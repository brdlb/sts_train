"""
Main script for training Perudo agents with parameter sharing and self-play.
"""

import os
import re
import glob
import threading
import contextlib
import shutil
import json
from io import StringIO
import numpy as np
from typing import List, Optional
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from typing import Any

from ..game.perudo_vec_env import PerudoMultiAgentVecEnv
from .config import Config, DEFAULT_CONFIG
from .opponent_pool import OpponentPool
from .rule_based_pool import RuleBasedOpponentPool
from ..agents.transformer_extractor import TransformerFeaturesExtractor
from .callbacks import (
    AdaptiveEntropyCallback,
    SelfPlayTrainingCallback,
    ModelUpdateProgressCallback,
    WinnerTrajectoryCollectorCallback,
    DateCheckpointCallback,
)
from .debug import DebugModeManager, get_debug_mode, set_debug_mode
from .utils import get_device, linear_schedule, find_latest_model, restore_model_from_opponent_pool

import sys
import torch
import logging

# Setup logger for this module
logger = logging.getLogger(__name__)

# --- Clean module from sys.modules to avoid runpy warning ---
if __name__ == "__main__":
    modname = __name__
    if modname in sys.modules:
        del sys.modules[modname]


class ActionMaskVecMonitor(VecMonitor):
    """
    VecMonitor wrapper that forwards action_masks method.
    
    VecMonitor doesn't automatically forward custom methods like action_masks
    from the underlying VecEnv. This wrapper ensures action_masks() is properly
    forwarded for use with MaskablePPO.
    """
    
    def __init__(self, venv: Any, filename: Optional[str] = None):
        """
        Initialize ActionMaskVecMonitor.
        
        Args:
            venv: The underlying VecEnv to wrap
            filename: Optional filename for monitoring
        """
        super().__init__(venv, filename)
        self._underlying_venv = venv
    
    def action_masks(self) -> np.ndarray:
        """
        Forward action_masks from underlying VecEnv.
        
        Returns:
            Boolean array of shape (num_envs, action_space.n) indicating valid actions
        """
        if hasattr(self._underlying_venv, 'action_masks'):
            return self._underlying_venv.action_masks()
        # Fallback: return all actions as valid
        return np.ones((self.num_envs, self.action_space.n), dtype=bool)
    
    def has_attr(self, attr_name: str) -> bool:
        """
        Check if attribute exists in wrapper or underlying VecEnv.
        
        Args:
            attr_name: Name of the attribute to check
            
        Returns:
            True if attribute exists, False otherwise
        """
        if hasattr(self, attr_name):
            return True
        return super().has_attr(attr_name)


class SelfPlayTraining:
    """
    Class for self-play training with parameter sharing and opponent pool.
    
    Features:
    - Parameter sharing: one PPO model for all agents
    - Opponent pool: sample opponents from pool
    - Batch normalization: normalize advantages globally
    - VecEnv: each environment = one table with 4 agents
    """
    
    def __init__(self, config: Config = DEFAULT_CONFIG):
        """
        Initialize training.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.num_players = config.game.num_players
        self.num_envs = config.training.num_envs
        # Track initial timesteps for continued training
        self.initial_timesteps = 0
        
        self._setup_directories()
        self._initialize_opponent_pools()
        self._create_environment()
        self._load_or_create_model()
        self._setup_statistics()
    
    def _setup_directories(self) -> None:
        """Create and configure directories."""
        # Ensure paths are absolute for consistency
        log_dir = os.path.abspath(self.config.training.log_dir)
        model_dir = os.path.abspath(self.config.training.model_dir)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Update config with absolute paths
        self.config.training.log_dir = log_dir
        self.config.training.model_dir = model_dir
    
    def _initialize_opponent_pools(self) -> None:
        """Initialize rule-based and RL opponent pools."""
        # Create opponent pool directory
        pool_dir = os.path.join(self.config.training.model_dir, "opponent_pool")
        # Use CPU for opponent models to avoid GPU overhead from single-observation predictions
        opponent_device = getattr(self.config.training, 'opponent_device', 'cpu')
        
        # Get max_history_length for use in pools and environment
        max_history_length = getattr(self.config.training, 'transformer_history_length', self.config.game.history_length)
        
        # Initialize opponent pools based on training mode
        self.training_mode = getattr(self.config.training, 'training_mode', 'selfplay')
        self.use_rule_based_opponents = getattr(self.config.training, 'use_rule_based_opponents', False)
        self.mixed_mode_ratio = getattr(self.config.training, 'mixed_mode_ratio', 0.5)
        
        # Create rule-based opponent pool if needed
        self.rule_based_pool = None
        if self.use_rule_based_opponents and self.training_mode in ('botplay', 'mixed'):
            bot_difficulty_distribution = getattr(
                self.config.training, 'bot_difficulty_distribution',
                {"EASY": 0.33, "MEDIUM": 0.34, "HARD": 0.33}
            )
            allowed_bot_personalities = getattr(
                self.config.training, 'allowed_bot_personalities', None
            )
            # Set statistics directory for rule-based bots
            statistics_dir = os.path.join(self.config.training.model_dir, "rule_based_pool")
            self.rule_based_pool = RuleBasedOpponentPool(
                max_quantity=self.config.game.max_quantity,
                max_players=self.config.game.max_players,
                max_history_length=max_history_length,
                difficulty_distribution=bot_difficulty_distribution,
                statistics_dir=statistics_dir,
                allowed_bot_personalities=allowed_bot_personalities,
            )
            if allowed_bot_personalities:
                logger.info(f"Initialized rule-based opponent pool with {len(self.rule_based_pool.bots)} bot personalities: {allowed_bot_personalities}")
            else:
                logger.info(f"Initialized rule-based opponent pool with {len(self.rule_based_pool.bots)} bot personalities")
        
        # Create RL opponent pool (for selfplay or mixed mode)
        self.opponent_pool = None
        if self.training_mode in ('selfplay', 'mixed'):
            self.opponent_pool = OpponentPool(
                pool_dir=pool_dir,
                max_pool_size=30,
                min_pool_size=20,
                keep_best=3,
                snapshot_freq=50000,
                opponent_device=opponent_device,
            )
        else:
            # In botplay mode, don't create RL opponent pool
            logger.info("Training mode: botplay - using only rule-based opponents")
    
    def _create_environment(self) -> None:
        """Create and wrap vectorized environment."""
        # Get max_history_length for use in environment
        max_history_length = getattr(self.config.training, 'transformer_history_length', self.config.game.history_length)
        
        # Create vectorized environment
        collect_trajectories = getattr(self.config.training, 'collect_trajectories', False)
        vec_env_raw = PerudoMultiAgentVecEnv(
            num_envs=self.num_envs,
            num_players=self.num_players,
            dice_per_player=self.config.game.dice_per_player,
            total_dice_values=self.config.game.total_dice_values,
            max_quantity=self.config.game.max_quantity,
            history_length=self.config.game.history_length,
            max_history_length=max_history_length,
            opponent_pool=self.opponent_pool,
            random_num_players=self.config.game.random_num_players,
            min_players=self.config.game.min_players,
            max_players=self.config.game.max_players,
            reward_config=self.config.reward,
            rule_based_pool=self.rule_based_pool,
            training_mode=self.training_mode,
            mixed_mode_ratio=self.mixed_mode_ratio,
            collect_trajectories=collect_trajectories,
        )
        
        # Wrap with ActionMaskVecMonitor for logging and action mask forwarding
        monitor_dir = os.path.join(self.config.training.log_dir, "monitor")
        os.makedirs(monitor_dir, exist_ok=True)
        vec_env_monitored = ActionMaskVecMonitor(vec_env_raw, monitor_dir)
        
        self.vec_env = vec_env_monitored
        
        # Store reference to original env for direct access if needed
        # VecMonitor should forward attributes, but we keep reference just in case
        self._vec_env_raw = vec_env_raw
    
    def _load_or_create_model(self) -> None:
        """Load existing model or create new one."""
        # Determine device for training (GPU with CPU fallback)
        device = get_device(self.config.training.device)
        
        # Create opponent pool directory for restore check
        pool_dir = os.path.join(self.config.training.model_dir, "opponent_pool")
        
        # Try to restore model from opponent pool if no model exists in root
        # This allows continuing training from opponent pool snapshots
        restored_model_path = restore_model_from_opponent_pool(
            self.config.training.model_dir,
            pool_dir
        )
        if restored_model_path:
            logger.info(f"Successfully restored model from opponent pool to: {restored_model_path}")
        
        # Try to find and load the latest saved model
        latest_model_path = find_latest_model(self.config.training.model_dir, additional_dirs=[pool_dir])
        
        if latest_model_path and os.path.exists(latest_model_path):
            # Load existing model for continued training
            logger.info(f"Found existing model: {latest_model_path}")
            logger.info("Loading model for continued training...")
            try:
                # Suppress SB3 wrapping messages
                with contextlib.redirect_stdout(StringIO()):
                    self.model = MaskablePPO.load(latest_model_path, env=self.vec_env)
                # Enable TensorBoard logging for loaded model
                # Ensure tensorboard_log is set to absolute path
                self.model.tensorboard_log = os.path.abspath(self.config.training.log_dir)
                
                # Update learning rate schedule for continued training
                # Analysis: Gentler decay to maintain learning capacity while ensuring stability
                # Final LR is 50% of initial (gentler decay for better exploration)
                initial_lr = self.config.training.learning_rate
                lr_schedule = linear_schedule(initial_lr, decay_ratio=0.5)
                self.model.learning_rate = lr_schedule
                final_lr = initial_lr * 0.5
                
                logger.info(f"Successfully loaded model from {latest_model_path}")
                logger.info(f"TensorBoard logging enabled: {self.model.tensorboard_log}")
                logger.info(f"Learning rate schedule updated: {initial_lr:.2e} -> {final_lr:.2e} (linear decay, 50% of initial)")
                
                # Get current timesteps from model (SB3 saves this in the model)
                self.initial_timesteps = self.model.num_timesteps if hasattr(self.model, 'num_timesteps') else 0
                
                # Extract step count from filename if possible (for verification)
                match = re.search(r"perudo_model_(\d+)_steps\.zip", os.path.basename(latest_model_path))
                if match:
                    filename_steps = int(match.group(1))
                    logger.info(f"Model filename indicates {filename_steps} steps. Model num_timesteps: {self.initial_timesteps}")
                    
                    # If model.num_timesteps doesn't match filename, use filename value
                    # This can happen if model was saved but num_timesteps wasn't properly saved
                    if self.initial_timesteps == 0 or abs(self.initial_timesteps - filename_steps) > 1000:
                        logger.warning(f"Model num_timesteps ({self.initial_timesteps}) doesn't match filename ({filename_steps})")
                        logger.warning(f"Using filename steps ({filename_steps}) as initial_timesteps")
                        self.initial_timesteps = filename_steps
                        # Try to set num_timesteps in model if possible (SB3 might not allow this directly)
                        # We'll rely on the corrected total_timesteps calculation instead
                else:
                    logger.info(f"Continuing training from {self.initial_timesteps} steps.")
                
                if self.initial_timesteps > 0:
                    logger.info(f"Training will continue from step {self.initial_timesteps:,}")
                else:
                    logger.warning("initial_timesteps is 0. Training will start from the beginning.")
            except Exception as e:
                logger.warning(f"Failed to load model from {latest_model_path}: {e}")
                logger.info("Creating new model instead...")
                # Fall through to create new model
                latest_model_path = None
        
        if latest_model_path is None or not os.path.exists(latest_model_path):
            # Create new PPO model with parameter sharing
            # One model for all agents (agent_id is in observation)
            # verbose=1 enables built-in progress bar showing timesteps and episode rewards
            logger.info("Creating new model from scratch...")
            
            # Build policy_kwargs with transformer extractor if not provided
            if self.config.training.policy_kwargs is None:
                # Get observation space from environment
                obs_space = self.vec_env.observation_space
                
                # Create policy_kwargs with transformer extractor
                policy_kwargs = dict(
                    features_extractor_class=TransformerFeaturesExtractor,
                    features_extractor_kwargs=dict(
                        features_dim=self.config.training.transformer_features_dim,
                        num_layers=self.config.training.transformer_num_layers,
                        num_heads=self.config.training.transformer_num_heads,
                        embed_dim=self.config.training.transformer_embed_dim,
                        dim_feedforward=self.config.training.transformer_dim_feedforward,
                        max_history_length=self.config.training.transformer_history_length,
                        max_quantity=self.config.game.max_quantity,
                        dropout=getattr(self.config.training, 'transformer_dropout', 0.1),
                    ),
                    # Enhanced network architectures for policy and value networks
                    # Value network receives 256-dim features, needs stronger architecture
                    # Format: list with dict specifying separate architectures for pi and vf
                    net_arch=[dict(pi=[256, 192, 128, 96], vf=[256, 192, 128, 96])],
                )
            else:
                policy_kwargs = self.config.training.policy_kwargs
            
            # Create learning rate schedule: linear decay with gentler rate
            # Analysis: Gentler decay to maintain learning capacity while ensuring stability
            # Final LR is 50% of initial (gentler decay for better exploration)
            initial_lr = self.config.training.learning_rate
            lr_schedule = linear_schedule(initial_lr, decay_ratio=0.5)
            final_lr = initial_lr * 0.5
            
            self.model = MaskablePPO(
                policy=self.config.training.policy,
                env=self.vec_env,
                device=device,
                policy_kwargs=policy_kwargs,
                learning_rate=lr_schedule,  # Use learning rate schedule
                n_steps=self.config.training.n_steps,
                batch_size=self.config.training.batch_size,
                n_epochs=self.config.training.n_epochs,
                gamma=self.config.training.gamma,
                gae_lambda=self.config.training.gae_lambda,
                clip_range=self.config.training.clip_range,
                ent_coef=self.config.training.ent_coef,
                vf_coef=self.config.training.vf_coef,
                max_grad_norm=self.config.training.max_grad_norm,
                verbose=self.config.training.verbose,  # 1 = progress bar, 0 = no output
                tensorboard_log=os.path.abspath(self.config.training.log_dir),  # Enable TensorBoard logging (absolute path)
            )
            
            logger.info(f"TensorBoard logging enabled: {self.model.tensorboard_log}")
            logger.info(f"Learning rate schedule: {initial_lr:.2e} -> {final_lr:.2e} (linear decay, 50% of initial)")
            
            # For new models, initial_timesteps is 0
            self.initial_timesteps = 0
            logger.info("Starting new training from step 0")
        
        # Update opponent pool with current model
        # VecMonitor forwards attributes, but we also set it on raw env for safety
        self.vec_env.current_model = self.model
        self._vec_env_raw.current_model = self.model
        # Update opponent models to use current model (for self-play mode)
        if hasattr(self._vec_env_raw, 'update_opponent_models_for_current_model'):
            self._vec_env_raw.update_opponent_models_for_current_model()
    
    def _setup_statistics(self) -> None:
        """Setup statistics tracking."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = [0] * self.num_players
        
    def train(self) -> None:
        """Main training loop."""
        device_str = str(self.model.device)
        logger.info("Starting training with parameter sharing")
        logger.info(f"Device: {device_str}")
        logger.info(f"Number of environments (tables): {self.num_envs}")
        logger.info(f"Number of players per table: {self.num_players}")
        
        # Start keyboard listener thread for debug mode toggle
        self._debug_manager = DebugModeManager()
        self._debug_manager.set(False)  # Enable debug mode by default
        #self._debug_manager.start_keyboard_listener()
        
        # Store debug mode flag in vec_env for access during training
        self._vec_env_raw.debug_mode = self._debug_manager.debug_mode_event
        
        # Adjust total_timesteps to account for already completed training
        # If model was loaded with initial_timesteps > 0, we need to train for
        # initial_timesteps + config.training.total_timesteps to get the correct
        # progress_remaining calculation for learning rate schedule
        config_total_timesteps = self.config.training.total_timesteps
        if self.initial_timesteps > 0:
            # Calculate total timesteps: train for additional config.training.total_timesteps steps
            # This ensures progress_remaining is calculated correctly
            total_timesteps = self.initial_timesteps + config_total_timesteps
            logger.info("Continuing training:")
            logger.info(f"  Initial timesteps: {self.initial_timesteps:,}")
            logger.info(f"  Additional timesteps: {config_total_timesteps:,}")
            logger.info(f"  Total timesteps: {total_timesteps:,}")
            logger.info(f"  Target progress: {self.initial_timesteps / total_timesteps * 100:.1f}% -> 100%")
        else:
            # New training: use config total_timesteps directly
            total_timesteps = config_total_timesteps
            logger.info("Starting new training:")
            logger.info(f"  Total timesteps: {total_timesteps:,}")
        
        # Calculate effective buffer size
        # Note: In multi-agent setup, only agent_id=0 is learning, but we collect data from all agents
        # The actual learning buffer size is n_steps * num_envs (one step per env per collection step)
        effective_buffer_size = self.config.training.n_steps * self.num_envs
        steps_per_env = self.config.training.n_steps
        num_batches_per_cycle = effective_buffer_size // self.config.training.batch_size
        gradient_updates_per_cycle = num_batches_per_cycle * self.config.training.n_epochs
        
        logger.info("Training configuration:")
        logger.info(f"  n_steps: {self.config.training.n_steps} (steps to collect before update)")
        logger.info(f"  batch_size: {self.config.training.batch_size} (mini-batch size)")
        logger.info(f"  n_epochs: {self.config.training.n_epochs} (epochs per update cycle)")
        logger.info(f"  Effective buffer size: {effective_buffer_size:,} samples")
        logger.info(f"  Steps per environment: {steps_per_env} (~{steps_per_env/1000:.1f}K steps/env)")
        logger.info(f"  Batches per cycle: {num_batches_per_cycle}")
        logger.info(f"  Gradient updates per cycle: {gradient_updates_per_cycle:,}")
        
        # Check that batch_size divides evenly
        if effective_buffer_size % self.config.training.batch_size != 0:
            logger.warning(f"Effective buffer size {effective_buffer_size} does not divide evenly "
                          f"by batch_size {self.config.training.batch_size}")
            logger.warning(f"Consider adjusting batch_size (current: {self.config.training.batch_size})")
            # Suggest divisors
            suggested_sizes = [s for s in [128, 256, 512, 1024] if effective_buffer_size % s == 0]
            suggested_str = ", ".join(map(str, suggested_sizes)) if suggested_sizes else "none found"
            logger.warning(f"  Suggested batch_size values that divide evenly: {suggested_str}")
        
        # Create callbacks
        callbacks = []
        

        
        # Adaptive entropy callback (if enabled)
        if getattr(self.config.training, 'adaptive_entropy', False):
            adaptive_entropy_callback = AdaptiveEntropyCallback(
                threshold_low=getattr(self.config.training, 'entropy_threshold_low', -3.5),
                threshold_high=getattr(self.config.training, 'entropy_threshold_high', -3.3),
                adjustment_rate=getattr(self.config.training, 'entropy_adjustment_rate', 0.008),
                max_ent_coef=getattr(self.config.training, 'entropy_max_coef', 0.25),
                verbose=self.config.training.verbose,
            )
            callbacks.append(adaptive_entropy_callback)
            logger.info("Adaptive entropy callback enabled:")
            logger.info(f"  Thresholds: low={adaptive_entropy_callback.threshold_low:.2f}, high={adaptive_entropy_callback.threshold_high:.2f}")
            logger.info(f"  Adjustment rate: {adaptive_entropy_callback.adjustment_rate:.4f}")
            logger.info(f"  Max ent_coef: {adaptive_entropy_callback.max_ent_coef:.4f}")
        
        # Self-play callback
        selfplay_callback = SelfPlayTrainingCallback(
            opponent_pool=self.opponent_pool,
            verbose=self.config.training.verbose,
            debug=False  # Enable debug mode by default
        )
        callbacks.append(selfplay_callback)
        
        # Model update progress callback
        model_update_callback = ModelUpdateProgressCallback(verbose=self.config.training.verbose)
        callbacks.append(model_update_callback)
        
        # Winner trajectory collector callback (only if enabled in config)
        collect_trajectories = getattr(self.config.training, 'collect_trajectories', False)
        if collect_trajectories:
            winner_trajectories_dir = os.path.join(self.config.training.model_dir, "winner_trajectories")
            winner_collector = WinnerTrajectoryCollectorCallback(
                data_dir=winner_trajectories_dir,
                verbose=self.config.training.verbose,
            )
            callbacks.append(winner_collector)
            logger.info(f"Winner trajectory collection enabled")
            logger.info(f"  Trajectories will be saved to: {winner_trajectories_dir}")
        
        # Checkpoint callback
        checkpoint_callback = DateCheckpointCallback(
            save_freq=self.config.training.save_freq,
            save_path=self.config.training.model_dir,
            name_prefix="perudo_model",
        )
        callbacks.append(checkpoint_callback)
        
        # Train model
        tb_log_name = self.config.training.tb_log_name or "perudo_training"
        
        # Verify that model.num_timesteps matches initial_timesteps
        # If not, we might need to manually set it (though SB3 should handle this)
        current_model_timesteps = self.model.num_timesteps if hasattr(self.model, 'num_timesteps') else 0
        if self.initial_timesteps > 0 and current_model_timesteps != self.initial_timesteps:
            logger.warning(f"Model num_timesteps ({current_model_timesteps}) doesn't match initial_timesteps ({self.initial_timesteps})")
            logger.warning("SB3 should handle this automatically, but progress calculations might be affected.")
            # Note: We can't directly set model.num_timesteps in SB3, but the learn() method
            # should respect the current value. The total_timesteps adjustment above should
            # ensure progress_remaining is calculated correctly.
        
        self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            callback=callbacks,
            progress_bar=True,  # Enable progress bar for better visibility
        )
        
        logger.info("Training completed!")
        
        # Save final model
        model_path = os.path.join(
            self.config.training.model_dir, "perudo_model_final.zip"
        )
        self.model.save(model_path)
        logger.info(f"Final model saved to {model_path}")
        
        # Log pool statistics (only if opponent_pool exists)
        if self.opponent_pool is not None:
            pool_stats = self.opponent_pool.get_statistics()
            logger.info(f"Opponent pool statistics: {pool_stats}")
        
    def save_model(self, path: str) -> None:
        """Save the current model."""
        self.model.save(path)
        
    def load_model(self, path: str) -> None:
        """Load a model."""
        # Suppress SB3 wrapping messages
        with contextlib.redirect_stdout(StringIO()):
            self.model = MaskablePPO.load(path, env=self.vec_env)
        self.vec_env.current_model = self.model
        # Update opponent models to use current model (for self-play mode)
        if hasattr(self._vec_env_raw, 'update_opponent_models_for_current_model'):
            self._vec_env_raw.update_opponent_models_for_current_model()


if __name__ == "__main__":
    from .cli import main
    main()
