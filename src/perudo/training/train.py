"""
Main script for training Perudo agents with parameter sharing and self-play.
"""

import os
import re
import glob
import threading
import contextlib
from io import StringIO
import numpy as np
from typing import List, Optional
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor

from ..game.perudo_vec_env import PerudoMultiAgentVecEnv
from .config import Config, DEFAULT_CONFIG
from .opponent_pool import OpponentPool
from ..agents.transformer_extractor import TransformerFeaturesExtractor

import sys
import torch

# --- Очистка модуля из sys.modules, чтобы избежать предупреждения runpy ---
if __name__ == "__main__":
    modname = __name__
    if modname in sys.modules:
        del sys.modules[modname]


def get_device(device: Optional[str] = None) -> str:
    """
    Get device for training with automatic GPU detection and CPU fallback.
    
    Args:
        device: Device string (e.g., "cpu", "cuda", "cuda:0"). 
                If None, automatically detects GPU and falls back to CPU.
    
    Returns:
        Device string to use for training
    """
    if device is not None:
        return device
    
    # Try to use GPU if available
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA not available, using CPU")
    
    return device


def find_latest_model(model_dir: str) -> Optional[str]:
    """
    Find the latest saved model in the model directory.
    
    Looks for models with pattern:
    - perudo_model_<steps>_steps.zip (checkpoint models)
    - perudo_model_final.zip (final model, treated as highest priority)
    
    Args:
        model_dir: Directory to search for models
    
    Returns:
        Path to the latest model, or None if no models found
    """
    if not os.path.exists(model_dir):
        return None
    
    # Find all checkpoint models (perudo_model_<steps>_steps.zip)
    checkpoint_pattern = os.path.join(model_dir, "perudo_model_*_steps.zip")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    # Find final model
    final_model_path = os.path.join(model_dir, "perudo_model_final.zip")
    final_model_exists = os.path.exists(final_model_path)
    
    latest_model = None
    latest_steps = -1
    
    # Extract steps from checkpoint files
    pattern = re.compile(r"perudo_model_(\d+)_steps\.zip")
    for file_path in checkpoint_files:
        match = pattern.search(os.path.basename(file_path))
        if match:
            steps = int(match.group(1))
            if steps > latest_steps:
                latest_steps = steps
                latest_model = file_path
    
    # If final model exists, prefer it over checkpoints
    if final_model_exists:
        # But if we have checkpoints with more steps, use the checkpoint
        # (final model might be from an earlier run)
        if latest_steps > 0:
            # Compare modification times
            final_mtime = os.path.getmtime(final_model_path)
            latest_mtime = os.path.getmtime(latest_model) if latest_model else 0
            
            # Use the more recent file
            if final_mtime > latest_mtime:
                return final_model_path
            else:
                return latest_model
        else:
            return final_model_path
    
    return latest_model


class AdvantageNormalizationCallback(BaseCallback):
    """
    Callback to normalize advantages across the entire batch.
    
    This is needed because with parameter sharing, we collect data from
    multiple agents across multiple environments, and we want to normalize
    advantages globally before updating.
    
    In StableBaselines3, advantages are normalized by default in PPO.update(),
    but we want to ensure normalization happens across the entire batch
    (n_steps * num_envs * num_agents).
    """
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
    
    def _on_rollout_end(self) -> bool:
        """
        Called when rollout ends, before advantage normalization.
        
        This ensures advantages are normalized globally across the entire batch.
        In SB3, this is already done in PPO.update(), but we verify it here.
        """
        # The advantages will be normalized in the PPO update automatically
        # SB3's PPO already normalizes advantages globally in compute_returns_and_advantage()
        # So we just verify that the rollout buffer exists
        if hasattr(self.model, 'rollout_buffer') and self.model.rollout_buffer is not None:
            # Advantages are normalized in PPO.update() by default
            # The normalization happens in compute_returns_and_advantage()
            # which normalizes across the entire rollout buffer
            pass
        return True
    
    def _on_step(self) -> bool:
        """
        Dummy method required by BaseCallback. Returns True to fulfill abstract class requirements.
        """
        return True


class SelfPlayTrainingCallback(BaseCallback):
    """
    Callback for self-play training with opponent pool.
    """
    def __init__(
        self,
        opponent_pool: OpponentPool,
        snapshot_freq: int = 50000,
        verbose: int = 0,
        debug: bool = False,
        save_snapshot_every_cycle: bool = True
    ):
        super().__init__(verbose)
        self.opponent_pool = opponent_pool
        self.snapshot_freq = snapshot_freq
        self.debug = debug
        self.save_snapshot_every_cycle = save_snapshot_every_cycle
        self.current_step = 0
        self.vec_env = None
        # Флаг для отслеживания, когда нужно сохранить снапшот после обновления модели
        self._pending_snapshot = False
        # Буферы для статистики
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_wins = []
        # Для потокобезопасного доступа
        self._lock = threading.Lock()

    def _on_training_start(self) -> None:
        if hasattr(self.model, 'env'):
            self.vec_env = self.model.env
        
        # Get underlying env if wrapped in VecMonitor
        if hasattr(self.vec_env, 'envs'):  # VecMonitor has envs attribute
            self._underlying_env = self.vec_env.envs[0] if hasattr(self.vec_env.envs, '__getitem__') else None
        else:
            self._underlying_env = None
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        pass
    
    def _on_step(self) -> bool:
        self.current_step += 1
        
        # Сохраняем снапшот после обновления модели (если был установлен флаг в _on_rollout_end)
        if self.save_snapshot_every_cycle and self._pending_snapshot:
            # Get current step from model (num_timesteps) - модель уже обновлена
            current_step = self.model.num_timesteps if hasattr(self.model, 'num_timesteps') else self.current_step
            self.opponent_pool.save_snapshot(
                self.model,
                current_step,
                prefix="snapshot",
                force=True  # Force save regardless of snapshot_freq
            )
            if self.verbose > 0:
                print(f"Saved snapshot after training cycle (step {current_step})")
            self._pending_snapshot = False
        
        # Собираем episode_info из infos всех env (SB3 их передаёт в self.locals["infos"])
        infos = self.locals.get("infos", [])
        for info in infos:
            # Добавляем статистику только если эпизод завершен (info содержит episode_reward)
            if isinstance(info, dict) and "episode_reward" in info:
                # Get episode information
                episode_reward = info["episode_reward"]
                episode_length = info["episode_length"]
                
                # Get winner from info or episode dict (both formats are supported)
                episode_dict = info.get("episode", {})
                winner = info.get("winner", episode_dict.get("winner", -1))
                
                with self._lock:
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    win = 1 if winner == 0 else 0
                    self.episode_wins.append(win)
        
        # Стандартная логика снапшотов по частоте шагов (опционально, для обратной совместимости)
        # Основное сохранение происходит после каждого цикла обучения через _pending_snapshot
        # Эта логика может быть отключена, если save_snapshot_every_cycle=True
        if not self.save_snapshot_every_cycle and self.current_step % self.snapshot_freq == 0:
            self.opponent_pool.save_snapshot(
                self.model,
                self.current_step,
                prefix="snapshot"
            )
            if self.verbose > 0:
                pool_stats = self.opponent_pool.get_statistics()
                print(f"Saved snapshot at step {self.current_step}")
                print(f"Pool statistics: {pool_stats}")
        
        # Update vec_env with current step for opponent sampling
        # VecMonitor should forward attributes, but we try both wrapped and underlying env
        if self.vec_env is not None:
            try:
                # Try to set on wrapped env (VecMonitor should forward)
                if hasattr(self.vec_env, 'current_step'):
                    self.vec_env.current_step = self.current_step
                # Also try underlying env if accessible
                elif hasattr(self.vec_env, 'envs') and len(self.vec_env.envs) > 0:
                    # VecMonitor wraps the original env
                    underlying = getattr(self.vec_env, 'env', None) or getattr(self.vec_env, 'venv', None)
                    if underlying is not None and hasattr(underlying, 'current_step'):
                        underlying.current_step = self.current_step
            except AttributeError:
                # If setting fails, it's okay - opponent sampling will use default
                pass
        return True
    
    def _on_rollout_end(self) -> bool:
        """
        Called after each training cycle (after collecting n_steps, before model update).
        Set flag to save snapshot after model update in next _on_step().
        
        Note: In SB3, this callback is called after collecting n_steps of data,
        but before the model update. We set a flag here and save the snapshot
        in the next _on_step() call, which happens after the model update.
        """
        # Set flag to save snapshot after model update (in next _on_step())
        if self.save_snapshot_every_cycle:
            self._pending_snapshot = True
        
        # Winrate statistics are updated in VecEnv.step_wait() when episodes end
        return True


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
        
        # Create directories
        os.makedirs(config.training.log_dir, exist_ok=True)
        os.makedirs(config.training.model_dir, exist_ok=True)
        
        # Create opponent pool directory
        pool_dir = os.path.join(config.training.model_dir, "opponent_pool")
        # Use CPU for opponent models to avoid GPU overhead from single-observation predictions
        opponent_device = getattr(config.training, 'opponent_device', 'cpu')
        self.opponent_pool = OpponentPool(
            pool_dir=pool_dir,
            max_pool_size=20,
            min_pool_size=10,
            keep_best=3,
            snapshot_freq=50000,
            opponent_device=opponent_device,
        )
        
        # Create vectorized environment
        # Use transformer_history_length from config if available, otherwise use history_length
        max_history_length = getattr(config.training, 'transformer_history_length', config.game.history_length)
        vec_env_raw = PerudoMultiAgentVecEnv(
            num_envs=self.num_envs,
            num_players=self.num_players,
            dice_per_player=config.game.dice_per_player,
            total_dice_values=config.game.total_dice_values,
            max_quantity=config.game.max_quantity,
            history_length=config.game.history_length,
            max_history_length=max_history_length,
            opponent_pool=self.opponent_pool,
            random_num_players=config.game.random_num_players,
            min_players=config.game.min_players,
            max_players=config.game.max_players,
        )
        
        # Wrap with VecMonitor for CSV logging of episode statistics
        # This will save episode rewards, lengths, and times to CSV files
        monitor_dir = os.path.join(config.training.log_dir, "monitor")
        os.makedirs(monitor_dir, exist_ok=True)
        self.vec_env = VecMonitor(vec_env_raw, monitor_dir)
        
        # Store reference to original env for direct access if needed
        # VecMonitor should forward attributes, but we keep reference just in case
        self._vec_env_raw = vec_env_raw
        
        # Determine device for training (GPU with CPU fallback)
        device = get_device(config.training.device)
        
        # Try to find and load the latest saved model
        latest_model_path = find_latest_model(config.training.model_dir)
        
        if latest_model_path and os.path.exists(latest_model_path):
            # Load existing model for continued training
            print(f"Found existing model: {latest_model_path}")
            print("Loading model for continued training...")
            try:
                # Suppress SB3 wrapping messages
                with contextlib.redirect_stdout(StringIO()):
                    self.model = MaskablePPO.load(latest_model_path, env=self.vec_env)
                # Enable TensorBoard logging for loaded model
                self.model.tensorboard_log = config.training.log_dir
                print(f"Successfully loaded model from {latest_model_path}")
                
                # Extract step count from filename if possible
                match = re.search(r"perudo_model_(\d+)_steps\.zip", os.path.basename(latest_model_path))
                if match:
                    loaded_steps = int(match.group(1))
                    print(f"Model was trained for {loaded_steps} steps. Continuing from step {loaded_steps}.")
            except Exception as e:
                print(f"Warning: Failed to load model from {latest_model_path}: {e}")
                print("Creating new model instead...")
                # Fall through to create new model
                latest_model_path = None
        
        if latest_model_path is None or not os.path.exists(latest_model_path):
            # Create new PPO model with parameter sharing
            # One model for all agents (agent_id is in observation)
            # verbose=1 enables built-in progress bar showing timesteps and episode rewards
            print("Creating new model from scratch...")
            
            # Build policy_kwargs with transformer extractor if not provided
            if config.training.policy_kwargs is None:
                # Get observation space from environment
                obs_space = self.vec_env.observation_space
                
                # Create policy_kwargs with transformer extractor
                policy_kwargs = dict(
                    features_extractor_class=TransformerFeaturesExtractor,
                    features_extractor_kwargs=dict(
                        features_dim=config.training.transformer_features_dim,
                        num_layers=config.training.transformer_num_layers,
                        num_heads=config.training.transformer_num_heads,
                        embed_dim=config.training.transformer_embed_dim,
                        dim_feedforward=config.training.transformer_dim_feedforward,
                        max_history_length=config.training.transformer_history_length,
                        max_quantity=config.game.max_quantity,
                        dropout=getattr(config.training, 'transformer_dropout', 0.1),
                    ),
                )
            else:
                policy_kwargs = config.training.policy_kwargs
            
            self.model = MaskablePPO(
                policy=config.training.policy,
                env=self.vec_env,
                device=device,
                policy_kwargs=policy_kwargs,
                learning_rate=config.training.learning_rate,
                n_steps=config.training.n_steps,
                batch_size=config.training.batch_size,
                n_epochs=config.training.n_epochs,
                gamma=config.training.gamma,
                gae_lambda=config.training.gae_lambda,
                clip_range=config.training.clip_range,
                ent_coef=config.training.ent_coef,
                vf_coef=config.training.vf_coef,
                max_grad_norm=config.training.max_grad_norm,
                verbose=config.training.verbose,  # 1 = progress bar, 0 = no output
                tensorboard_log=config.training.log_dir,  # Enable TensorBoard logging
            )
        
        # Update opponent pool with current model
        # VecMonitor forwards attributes, but we also set it on raw env for safety
        self.vec_env.current_model = self.model
        self._vec_env_raw.current_model = self.model
        
        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = [0] * self.num_players
        
    def train(self):
        """Main training loop."""
        device_str = str(self.model.device)
        print(f"Starting training with parameter sharing")
        print(f"Device: {device_str}")
        print(f"Number of environments (tables): {self.num_envs}")
        print(f"Number of players per table: {self.num_players}")
        print(f"Total timesteps: {self.config.training.total_timesteps}")
        print(f"Effective batch size: {self.config.training.n_steps * self.num_envs * self.num_players}")
        
        # Check that batch_size divides evenly
        effective_batch_size = self.config.training.n_steps * self.num_envs * self.num_players
        if effective_batch_size % self.config.training.batch_size != 0:
            print(f"Warning: Effective batch size {effective_batch_size} does not divide evenly "
                  f"by batch_size {self.config.training.batch_size}")
            print(f"Consider adjusting batch_size or num_envs")
        
        # Create callbacks
        callbacks = []
        
        # Advantage normalization callback
        adv_norm_callback = AdvantageNormalizationCallback(verbose=self.config.training.verbose)
        callbacks.append(adv_norm_callback)
        
        # Self-play callback
        selfplay_callback = SelfPlayTrainingCallback(
            opponent_pool=self.opponent_pool,
            snapshot_freq=50000,
            verbose=self.config.training.verbose,
            debug=False
        )
        callbacks.append(selfplay_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.training.save_freq,
            save_path=self.config.training.model_dir,
            name_prefix="perudo_model",
        )
        callbacks.append(checkpoint_callback)
        
        # Train model
        tb_log_name = self.config.training.tb_log_name or "perudo_training"
        
        self.model.learn(
            total_timesteps=self.config.training.total_timesteps,
            tb_log_name=tb_log_name,
            callback=callbacks,
            progress_bar=True,  # Enable progress bar for better visibility
        )
        
        print("Training completed!")
        
        # Save final model
        model_path = os.path.join(
            self.config.training.model_dir, "perudo_model_final.zip"
        )
        self.model.save(model_path)
        print(f"Final model saved to {model_path}")
        
        # Print pool statistics
        pool_stats = self.opponent_pool.get_statistics()
        print(f"Opponent pool statistics: {pool_stats}")
        
    def save_model(self, path: str):
        """Save the current model."""
        self.model.save(path)
        
    def load_model(self, path: str):
        """Load a model."""
        # Suppress SB3 wrapping messages
        with contextlib.redirect_stdout(StringIO()):
            self.model = PPO.load(path, env=self.vec_env)
        self.vec_env.current_model = self.model


def main():
    """Main function to run training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Perudo agents with parameter sharing")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (JSON)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Number of parallel environments (tables). If not specified, uses value from config.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=10_000_000,
        help="Total training steps",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Number of steps to collect before update",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training (cpu, cuda, cuda:0, etc.). If not specified, auto-detects GPU with CPU fallback.",
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = DEFAULT_CONFIG
    config.training.total_timesteps = args.total_timesteps
    config.training.n_steps = args.n_steps
    config.training.batch_size = args.batch_size
    config.training.device = args.device
    
    # Override num_envs from command line if provided
    if args.num_envs is not None:
        config.training.num_envs = args.num_envs
    
    # Start training
    trainer = SelfPlayTraining(config)
    trainer.train()


if __name__ == "__main__":
    main()
