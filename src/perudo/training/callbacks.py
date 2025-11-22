"""
Training callbacks for Perudo agent training.

Contains callback classes for:
- Advantage normalization
- Adaptive entropy coefficient adjustment
- Self-play training with opponent pool
- Model update progress tracking
- Winner trajectory collection for imitation learning
"""

import threading
import time
import os
import pickle
import numpy as np
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
from stable_baselines3.common.callbacks import BaseCallback

from .opponent_pool import OpponentPool

# Setup logger for this module
logger = logging.getLogger(__name__)





class AdaptiveEntropyCallback(BaseCallback):
    """
    Callback to adaptively adjust entropy coefficient based on entropy loss.
    
    Monitors entropy loss and adjusts ent_coef to maintain exploration:
    - Increases ent_coef when entropy is too low (policy too deterministic)
    - Decreases ent_coef when entropy is too high (policy too random)
    """
    
    def __init__(
        self,
        threshold_low: float = -3.4,
        threshold_high: float = -3.2,
        adjustment_rate: float = 0.01,
        min_ent_coef: float = 0.01,
        max_ent_coef: float = 0.5,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.adjustment_rate = adjustment_rate
        self.min_ent_coef = min_ent_coef
        self.max_ent_coef = max_ent_coef
        self.last_entropy = None
        self.initial_ent_coef = None
    
    def _on_training_start(self) -> None:
        """Store initial ent_coef value."""
        if hasattr(self.model, 'ent_coef'):
            self.initial_ent_coef = self.model.ent_coef
            
            if self.verbose > 0:
                logger.info("AdaptiveEntropyCallback initialized:")
                logger.info(f"  Initial ent_coef: {self.initial_ent_coef:.4f}")
                logger.info(f"  Thresholds: low={self.threshold_low:.2f}, high={self.threshold_high:.2f}")
                logger.info(f"  Adjustment rate: {self.adjustment_rate:.4f}")
    
    def _on_rollout_end(self) -> bool:
        """
        Called after model update. Adjust ent_coef based on entropy loss.
        """
        # Try to get entropy_loss from logger
        if hasattr(self, 'logger') and self.logger is not None:
            try:
                entropy_loss = None
                if hasattr(self.logger, 'name_to_value'):
                    entropy_loss = self.logger.name_to_value.get('train/entropy_loss', None)
                
                if entropy_loss is not None:
                    self.last_entropy = entropy_loss
                    old_ent_coef = self.model.ent_coef
                    new_ent_coef = old_ent_coef
                    
                    if entropy_loss < self.threshold_low:
                        new_ent_coef = min(old_ent_coef + self.adjustment_rate, self.max_ent_coef)
                        if new_ent_coef != old_ent_coef and self.verbose > 0:
                            logger.info(f"Entropy too low ({entropy_loss:.4f} < {self.threshold_low:.2f}), "
                                       f"increasing ent_coef: {old_ent_coef:.4f} -> {new_ent_coef:.4f}")
                    elif entropy_loss > self.threshold_high:
                        new_ent_coef = max(old_ent_coef - self.adjustment_rate, self.min_ent_coef)
                        if new_ent_coef != old_ent_coef and self.verbose > 0:
                            logger.info(f"Entropy too high ({entropy_loss:.4f} > {self.threshold_high:.2f}), "
                                       f"decreasing ent_coef: {old_ent_coef:.4f} -> {new_ent_coef:.4f}")
                    
                    if new_ent_coef != old_ent_coef:
                        self.model.ent_coef = new_ent_coef
                        self.logger.record("custom/ent_coef", new_ent_coef)
                        self.logger.record("custom/entropy_loss", entropy_loss)
                        self.logger.record("custom/ent_coef_adjustment", new_ent_coef - old_ent_coef)
                
            except Exception as e:
                if self.verbose > 0:
                    logger.warning(f"Failed to adjust entropy coefficient: {e}")
        
        return True
    
    def _on_step(self) -> bool:
        return True


class SelfPlayTrainingCallback(BaseCallback):
    """
    Callback for self-play training with opponent pool.
    """
    def __init__(
        self,
        opponent_pool: Optional[OpponentPool],
        verbose: int = 0,
        debug: bool = False,
        save_snapshot_every_cycle: bool = True,
        snapshot_cycle_freq: int = 1
    ):
        super().__init__(verbose)
        self.opponent_pool = opponent_pool
        self.debug = debug
        self.save_snapshot_every_cycle = save_snapshot_every_cycle
        self.snapshot_cycle_freq = snapshot_cycle_freq
        self.training_cycle_count = 0
        self.vec_env = None
        self._pending_snapshot = False
        
        # Buffers for statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_wins = []
        
        # Statistics for botplay/selfplay win ratio
        self.botplay_wins = 0
        self.botplay_episodes = 0
        self.selfplay_wins = 0
        self.selfplay_episodes = 0

    def _on_training_start(self) -> None:
        if hasattr(self.model, 'env'):
            self.vec_env = self.model.env
        
        if self.debug:
            debug_event = threading.Event()
            debug_event.set()
            
            if hasattr(self.vec_env, 'venv') and hasattr(self.vec_env.venv, 'debug_mode'):
                self.vec_env.venv.debug_mode = debug_event
            elif hasattr(self.vec_env, 'debug_mode'):
                self.vec_env.debug_mode = debug_event
            
            logger.info("[DEBUG MODE ENABLED] Training will pause after each action.")
    
    def _on_step(self) -> bool:
        # Save snapshot after model update (if flag was set in _on_rollout_end)
        if self.save_snapshot_every_cycle and self._pending_snapshot and self.opponent_pool is not None:
            snapshot_step = self.num_timesteps
            self.opponent_pool.save_snapshot(
                self.model,
                snapshot_step,
                prefix="snapshot",
                force=True
            )
            if self.verbose > 0:
                logger.info(f"Saved snapshot after training cycle {self.training_cycle_count} (step {snapshot_step:,})")
            self._pending_snapshot = False
        
        # Collect episode_info
        infos = self.locals.get("infos", [])
        for info in infos:
            if isinstance(info, dict) and "episode_reward" in info:
                episode_reward = info["episode_reward"]
                episode_length = info["episode_length"]
                
                episode_dict = info.get("episode", {})
                winner = info.get("winner", episode_dict.get("winner", -1))
                episode_mode = info.get("episode_mode", None)
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                win = 1 if winner == 0 else 0
                self.episode_wins.append(win)
                
                if episode_mode == "botplay":
                    self.botplay_episodes += 1
                    if win == 1:
                        self.botplay_wins += 1
                elif episode_mode == "selfplay":
                    self.selfplay_episodes += 1
                    if win == 1:
                        self.selfplay_wins += 1
        
        # Update vec_env with current step for opponent sampling
        if self.vec_env is not None:
            try:
                step_for_sampling = self.num_timesteps
                if hasattr(self.vec_env, 'current_step'):
                    self.vec_env.current_step = step_for_sampling
                elif hasattr(self.vec_env, 'envs') and len(self.vec_env.envs) > 0:
                    underlying = getattr(self.vec_env, 'env', None) or getattr(self.vec_env, 'venv', None)
                    if underlying is not None and hasattr(underlying, 'current_step'):
                        underlying.current_step = step_for_sampling
            except AttributeError:
                pass
        return True
    
    def _on_rollout_end(self) -> bool:
        self.training_cycle_count += 1
        
        if self.verbose > 0:
            logger.info(f"✓ Training cycle {self.training_cycle_count} completed! Steps: {self.num_timesteps}")
        
        if self.save_snapshot_every_cycle and self.opponent_pool is not None and self.training_cycle_count % self.snapshot_cycle_freq == 0:
            self._pending_snapshot = True
        
        # Log statistics
        if len(self.episode_rewards) > 0:
            mean_reward = np.mean(self.episode_rewards)
            mean_length = np.mean(self.episode_lengths)
            win_rate = np.mean(self.episode_wins) if len(self.episode_wins) > 0 else 0.0
            
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.record("custom/mean_episode_reward", mean_reward)
                self.logger.record("custom/mean_episode_length", mean_length)
                self.logger.record("custom/win_rate", win_rate)
                self.logger.record("custom/total_episodes", len(self.episode_rewards))
                
                if self.opponent_pool is not None:
                    pool_stats = self.opponent_pool.get_statistics()
                    if pool_stats:
                        self.logger.record("custom/opponent_pool_size", pool_stats.get("pool_size", 0))
                        self.logger.record("custom/opponent_pool_avg_winrate", pool_stats.get("average_winrate", 0.0))
                
                # Log mixed mode stats
                if self.botplay_episodes > 0:
                    self.logger.record("custom/botplay_winrate", self.botplay_wins / self.botplay_episodes)
                if self.selfplay_episodes > 0:
                    self.logger.record("custom/selfplay_winrate", self.selfplay_wins / self.selfplay_episodes)
            
            # Keep buffer size limited
            max_buffer_size = 100
            if len(self.episode_rewards) > max_buffer_size:
                self.episode_rewards = self.episode_rewards[-max_buffer_size:]
                self.episode_lengths = self.episode_lengths[-max_buffer_size:]
                self.episode_wins = self.episode_wins[-max_buffer_size:]
        
        return True


class ModelUpdateProgressCallback(BaseCallback):
    """
    Callback для отслеживания прогресса обновления модели.
    
    Логирует начало и завершение обновления модели, чтобы пользователь
    видел, что процесс не завис, а активно обновляет модель.
    
    После сбора данных (rollout) и вывода статистики, SB3 вызывает
    model.update(), который выполняет обучение на собранных данных.
    Этот callback показывает, что обновление началось и когда оно завершилось.
    """
    
    def __init__(self, verbose: int = 0):
        """
        Initialize model update progress callback.
        
        Args:
            verbose: Verbosity level (0 = silent, 1 = info, 2 = debug)
        """
        super().__init__(verbose)
        self.update_start_time = None
        self.last_timesteps = 0
        self.n_epochs = None
        self.batch_size = None
        self.n_steps = None
        self.num_envs = None
    
    def _on_training_start(self):
        """Инициализация при начале обучения."""
        self.n_epochs = getattr(self.model, 'n_epochs', 10)
        self.batch_size = getattr(self.model, 'batch_size', 1536)
        self.n_steps = getattr(self.model, 'n_steps', 8192)
        # Получаем количество окружений из env
        if hasattr(self.model, 'env') and hasattr(self.model.env, 'num_envs'):
            self.num_envs = self.model.env.num_envs
        else:
            self.num_envs = 1
        
        # Вычисляем количество батчей и обновлений
        effective_buffer_size = self.n_steps * self.num_envs
        num_batches = effective_buffer_size // self.batch_size
        total_updates = num_batches * self.n_epochs
        
        if self.verbose > 0:
            logger.info("ModelUpdateProgressCallback initialized:")
            logger.info(f"  n_epochs: {self.n_epochs}")
            logger.info(f"  batch_size: {self.batch_size}")
            logger.info(f"  n_steps: {self.n_steps}")
            logger.info(f"  num_envs: {self.num_envs}")
            logger.info(f"  Effective buffer size: {effective_buffer_size:,}")
            logger.info(f"  Batches per epoch: {num_batches}")
            logger.info(f"  Total gradient updates per cycle: {total_updates:,}")
    
    def _on_rollout_end(self) -> bool:
        """
        Вызывается после сбора данных, перед обновлением модели.
        Логируем начало обновления.
        """
        self.update_start_time = time.time()
        self.last_timesteps = self.model.num_timesteps if hasattr(self.model, 'num_timesteps') else 0
        
        if self.verbose > 0:
            effective_buffer_size = self.n_steps * self.num_envs
            num_batches = effective_buffer_size // self.batch_size
            total_updates = num_batches * self.n_epochs
            
            logger.info("")
            logger.info("🔄 Starting model update...")
            logger.info(f"   Training on {effective_buffer_size:,} samples")
            logger.info(f"   {self.n_epochs} epochs × {num_batches} batches = {total_updates:,} gradient updates")
            logger.info("   This may take a while, please wait...")
            logger.info("")
        
        return True
    
    def _on_step(self) -> bool:
        """
        Вызывается после обновления модели (в начале следующего цикла сбора данных).
        Логируем завершение обновления.
        """
        if self.update_start_time is not None:
            update_duration = time.time() - self.update_start_time
            
            current_timesteps = self.model.num_timesteps if hasattr(self.model, 'num_timesteps') else 0
            timesteps_increased = current_timesteps - self.last_timesteps
            
            if self.verbose > 0:
                logger.info("")
                logger.info("✅ Model update completed!")
                logger.info(f"   Duration: {update_duration:.2f} seconds")
                logger.info(f"   Timesteps increased: {timesteps_increased:,}")
                logger.info(f"   Current total timesteps: {current_timesteps:,}")
                logger.info("")
            
            self.update_start_time = None
        
        return True
    
    def _on_rollout_start(self) -> bool:
        """
        Вызывается в начале сбора данных (после обновления модели).
        Это дополнительная точка для логирования, если нужно.
        """
        return True


class WinnerTrajectoryCollectorCallback(BaseCallback):
    """
    Callback to collect episode trajectories from winners in botplay mode.
    
    Collects observations, actions, and rewards for the winning player
    in each episode. Data is saved to disk for later use in imitation learning.
    """
    
    def __init__(
        self,
        data_dir: str = "collected_trajectories",
        verbose: int = 0,
        min_episode_length: int = 0,  # Minimum episode length to save
        max_episode_length: Optional[int] = None,  # Maximum episode length to save
    ):
        """
        Initialize winner trajectory collector.
        
        Args:
            data_dir: Directory to save collected trajectories
            verbose: Verbosity level
            min_episode_length: Minimum episode length to save (filter short episodes)
            max_episode_length: Maximum episode length to save (filter long episodes)
        """
        super().__init__(verbose)
        self.data_dir = data_dir
        self.min_episode_length = min_episode_length
        self.max_episode_length = max_episode_length
        
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Statistics
        self.trajectories_collected = 0
        self.trajectories_filtered = 0
        self.vec_env = None
        
    def _on_training_start(self) -> None:
        """Get reference to vec_env."""
        if hasattr(self.model, 'env'):
            self.vec_env = self.model.env
            # Get underlying vec_env if wrapped in VecMonitor
            if hasattr(self.vec_env, '_vec_env_raw'):
                self.vec_env = self.vec_env._vec_env_raw
            elif hasattr(self.vec_env, 'venv'):
                self.vec_env = self.vec_env.venv
    
    def _on_step(self) -> bool:
        """Check for completed episodes and collect winner trajectories."""
        # Collect episode_info from all env infos (SB3 passes them in self.locals["infos"])
        infos = self.locals.get("infos", [])
        
        if not infos:
            return True
        
        # Process each environment
        for env_idx, info in enumerate(infos):
            # Check if episode ended
            episode_ended = False
            winner = None
            episode_length = None
            
            # Check for episode completion
            if isinstance(info, dict):
                # Check episode dict (VecMonitor format)
                episode_dict = info.get("episode", {})
                if episode_dict and "r" in episode_dict:
                    episode_ended = True
                    winner = info.get("winner") or episode_dict.get("winner", None)
                    episode_length = episode_dict.get("l", None)
                # Also check game_over flag (direct format)
                elif info.get("game_over", False):
                    episode_ended = True
                    winner = info.get("winner", None)
                    episode_length = info.get("episode_length", info.get("_episode_length_raw", None))
            
            if episode_ended and winner is not None and winner >= 0:
                self._collect_winner_trajectory(env_idx, winner, info, episode_length)
        
        return True
    
    def _collect_winner_trajectory(
        self, 
        env_idx: int, 
        winner: int, 
        info: Dict[str, Any],
        episode_length: Optional[int] = None
    ) -> None:
        """Collect and save trajectory for the winning player."""
        if self.vec_env is None or not hasattr(self.vec_env, 'envs') or env_idx >= len(self.vec_env.envs):
            return
        
        env = self.vec_env.envs[env_idx]
        
        # Get trajectory for winner
        trajectory_data = env.get_player_trajectory(winner)
        if not trajectory_data:
            return
        
        # Filter by episode length if specified
        actual_length = len(trajectory_data) if episode_length is None else episode_length
        if self.min_episode_length > 0 and actual_length < self.min_episode_length:
            self.trajectories_filtered += 1
            return
        
        if self.max_episode_length is not None and actual_length > self.max_episode_length:
            self.trajectories_filtered += 1
            return
        
        # Extract observations, actions, rewards, and action masks
        observations = [entry["observation"] for entry in trajectory_data]
        actions = [entry["action"] for entry in trajectory_data]
        rewards = [entry["reward"] for entry in trajectory_data]
        action_masks = [entry.get("action_mask", None) for entry in trajectory_data]
        
        # Prepare trajectory for saving
        trajectory = {
            "winner": winner,
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "action_masks": action_masks,
            "episode_info": info,
            "num_players": env.num_players,
            "episode_length": actual_length,
        }
        
        # Save trajectory to file
        import time
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
        filename = f"trajectory_{timestamp}_{env_idx}_{winner}.pkl"
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(trajectory, f)
            
            self.trajectories_collected += 1
            
            if self.verbose > 0 and self.trajectories_collected % 10 == 0:
                logger.info(f"Collected {self.trajectories_collected} trajectories "
                          f"(filtered: {self.trajectories_filtered})")
        
        except Exception as e:
            logger.warning(f"Failed to save trajectory for env {env_idx}, winner {winner}: {e}")
    
    def _on_training_end(self) -> None:
        """Log final statistics."""
        if self.verbose > 0:
            logger.info(f"WinnerTrajectoryCollectorCallback finished:")
            logger.info(f"  Total trajectories collected: {self.trajectories_collected}")
            logger.info(f"  Total trajectories filtered: {self.trajectories_filtered}")
            logger.info(f"  Data directory: {self.data_dir}")


class DateCheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    The filename includes the date instead of step count.
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.save_path, f"{self.name_prefix}_{date_str}")
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True



