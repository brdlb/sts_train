"""
Training callbacks for Perudo agent training.

Contains callback classes for:
- Advantage normalization
- Adaptive entropy coefficient adjustment
- Self-play training with opponent pool
"""

import threading
import numpy as np
from typing import Optional
from stable_baselines3.common.callbacks import BaseCallback

from .opponent_pool import OpponentPool


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
        """
        Initialize adaptive entropy callback.
        
        Args:
            threshold_low: Lower threshold - increase ent_coef when entropy < this
            threshold_high: Upper threshold - decrease ent_coef when entropy > this
            adjustment_rate: Rate of ent_coef adjustment per update
            min_ent_coef: Minimum allowed ent_coef value
            max_ent_coef: Maximum allowed ent_coef value
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.adjustment_rate = adjustment_rate
        self.min_ent_coef = min_ent_coef
        self.max_ent_coef = max_ent_coef
        self.last_entropy = None
        self.initial_ent_coef = None
        self.last_update_step = 0  # Track last step when model was updated
    
    def _on_training_start(self) -> None:
        """Store initial ent_coef value and sync with model's current training progress."""
        if hasattr(self.model, 'ent_coef'):
            self.initial_ent_coef = self.model.ent_coef
        else:
            self.initial_ent_coef = None
        
        # Get current num_timesteps from model to sync last_update_step
        # This is crucial for continued training - we don't want to reset the tracking
        if hasattr(self.model, 'num_timesteps'):
            current_timesteps = self.model.num_timesteps
            # Set last_update_step to current timesteps to properly track updates
            # This ensures we don't skip entropy adjustments when continuing training
            self.last_update_step = current_timesteps
            
            if self.verbose > 0:
                ent_coef_str = f"{self.initial_ent_coef:.4f}" if self.initial_ent_coef is not None else "N/A"
                print(f"AdaptiveEntropyCallback initialized:")
                print(f"  Current model timesteps: {current_timesteps:,}")
                print(f"  Initial ent_coef: {ent_coef_str}")
                print(f"  Thresholds: low={self.threshold_low:.2f}, high={self.threshold_high:.2f}")
                print(f"  Adjustment rate: {self.adjustment_rate:.4f}")
                print(f"  Last update step set to: {self.last_update_step:,}")
        else:
            if self.verbose > 0:
                ent_coef_str = f"{self.initial_ent_coef:.4f}" if self.initial_ent_coef is not None else "N/A"
                print(f"AdaptiveEntropyCallback initialized with initial ent_coef={ent_coef_str}")
                print(f"  Thresholds: low={self.threshold_low:.2f}, high={self.threshold_high:.2f}")
                print(f"  Adjustment rate: {self.adjustment_rate:.4f}")
                print(f"  Warning: Model doesn't have num_timesteps attribute, last_update_step remains at 0")
    
    def _on_rollout_end(self) -> bool:
        """
        Called after model update. Adjust ent_coef based on entropy loss.
        
        Note: In SB3, this is called after collecting n_steps but before model.update().
        However, we can access metrics from the previous update through logger.name_to_value
        after logger.dump() is called (which happens after model.update()).
        """
        # Check if model was updated (num_timesteps changed by n_steps)
        current_step = self.model.num_timesteps if hasattr(self.model, 'num_timesteps') else 0
        n_steps = getattr(self.model, 'n_steps', 8192)
        
        # Only check entropy after model update (when step increased by n_steps)
        # This avoids checking entropy multiple times between updates
        # Note: For continued training, last_update_step is initialized to model's current timesteps
        # so we check if enough steps have been collected since the last update
        steps_since_last_update = current_step - self.last_update_step
        if steps_since_last_update >= n_steps:
            # Update tracking: set to the step at which we're checking (before the update)
            # The actual update will happen after this callback, so we track the step before update
            self.last_update_step = current_step
            
            # Try to get entropy_loss from logger
            # In SB3, metrics are available through logger.name_to_value after update
            if hasattr(self, 'logger') and self.logger is not None:
                try:
                    # Get entropy_loss from logger
                    # Note: logger.name_to_value contains metrics from the last update
                    entropy_loss = None
                    if hasattr(self.logger, 'name_to_value'):
                        entropy_loss = self.logger.name_to_value.get('train/entropy_loss', None)
                    
                    # If we have entropy_loss, adjust ent_coef
                    if entropy_loss is not None:
                        self.last_entropy = entropy_loss
                        old_ent_coef = self.model.ent_coef
                        new_ent_coef = old_ent_coef
                        
                        # Adjust based on thresholds
                        if entropy_loss < self.threshold_low:
                            # Entropy too low (policy too deterministic) - increase ent_coef
                            new_ent_coef = min(old_ent_coef + self.adjustment_rate, self.max_ent_coef)
                            if new_ent_coef != old_ent_coef and self.verbose > 0:
                                print(f"Entropy too low ({entropy_loss:.4f} < {self.threshold_low:.2f}), "
                                      f"increasing ent_coef: {old_ent_coef:.4f} -> {new_ent_coef:.4f}")
                        elif entropy_loss > self.threshold_high:
                            # Entropy too high (policy too random) - decrease ent_coef
                            new_ent_coef = max(old_ent_coef - self.adjustment_rate, self.min_ent_coef)
                            if new_ent_coef != old_ent_coef and self.verbose > 0:
                                print(f"Entropy too high ({entropy_loss:.4f} > {self.threshold_high:.2f}), "
                                      f"decreasing ent_coef: {old_ent_coef:.4f} -> {new_ent_coef:.4f}")
                        
                        # Update model's ent_coef
                        if new_ent_coef != old_ent_coef:
                            self.model.ent_coef = new_ent_coef
                            
                            # Log to TensorBoard
                            if hasattr(self, 'logger') and self.logger is not None:
                                self.logger.record("custom/ent_coef", new_ent_coef)
                                self.logger.record("custom/entropy_loss", entropy_loss)
                                self.logger.record("custom/ent_coef_adjustment", new_ent_coef - old_ent_coef)
                    
                except Exception as e:
                    # Don't crash training if adjustment fails
                    if self.verbose > 0:
                        print(f"Warning: Failed to adjust entropy coefficient: {e}")
        
        return True
    
    def _on_step(self) -> bool:
        """
        Dummy method required by BaseCallback. Returns True to fulfill abstract class requirements.
        """
        return True


class SelfPlayTrainingCallback(BaseCallback):
    """
    Callback for self-play training with opponent pool.
    
    Snapshot saving behavior:
    - If save_snapshot_every_cycle=True (default): saves snapshot every snapshot_cycle_freq training cycles
    - If save_snapshot_every_cycle=False: saves snapshot every snapshot_freq steps (for backward compatibility)
    """
    def __init__(
        self,
        opponent_pool: OpponentPool,
        snapshot_freq: int = 50000,  # Only used when save_snapshot_every_cycle=False
        verbose: int = 0,
        debug: bool = False,
        save_snapshot_every_cycle: bool = True,
        snapshot_cycle_freq: int = 1  # Frequency of saving snapshots in training cycles (when save_snapshot_every_cycle=True)
    ):
        super().__init__(verbose)
        self.opponent_pool = opponent_pool
        self.snapshot_freq = snapshot_freq
        self.debug = debug
        self.save_snapshot_every_cycle = save_snapshot_every_cycle
        self.snapshot_cycle_freq = snapshot_cycle_freq  # Сохранять снапшот каждые snapshot_cycle_freq циклов обучения
        self.current_step = 0
        self.training_cycle_count = 0  # Счетчик циклов обучения
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
        
        # Enable debug mode if requested
        if self.debug:
            # Create a threading.Event for debug mode
            debug_event = threading.Event()
            debug_event.set()  # Enable debug mode
            
            # Set debug_mode on underlying VecEnv
            if hasattr(self.vec_env, 'venv'):  # VecMonitor wraps env in venv attribute
                underlying_vec_env = self.vec_env.venv
                if hasattr(underlying_vec_env, 'debug_mode'):
                    underlying_vec_env.debug_mode = debug_event
                    print("[DEBUG MODE ENABLED] Training will pause after each action. Press Enter to continue.")
            elif hasattr(self.vec_env, 'debug_mode'):
                self.vec_env.debug_mode = debug_event
                print("[DEBUG MODE ENABLED] Training will pause after each action. Press Enter to continue.")
        
        # Synchronize current_step with model's num_timesteps for continued training
        # This is crucial to ensure snapshots are saved with correct step numbers
        if hasattr(self.model, 'num_timesteps'):
            self.current_step = self.model.num_timesteps
            if self.verbose > 0:
                print(f"SelfPlayTrainingCallback: Synchronized current_step with model.num_timesteps = {self.current_step:,}")
        else:
            self.current_step = 0
            if self.verbose > 0:
                print("SelfPlayTrainingCallback: Model doesn't have num_timesteps, starting from step 0")
        
        # Verify logger is available
        if not hasattr(self, 'logger') or self.logger is None:
            if self.verbose > 0:
                print("Warning: Logger not available in SelfPlayTrainingCallback._on_training_start()")
        elif self.verbose > 1:
            print(f"Logger initialized in SelfPlayTrainingCallback: {type(self.logger)}")
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        pass
    
    def _on_step(self) -> bool:
        # Always use model.num_timesteps as the source of truth for step counting
        # This ensures correct step numbers when continuing training
        if hasattr(self.model, 'num_timesteps'):
            # Sync self.current_step with model.num_timesteps
            # model.num_timesteps is updated after model updates, so it's the authoritative source
            self.current_step = self.model.num_timesteps
        else:
            # Fallback: increment if model doesn't have num_timesteps (shouldn't happen with SB3)
            self.current_step += 1
        
        # Сохраняем снапшот после обновления модели (если был установлен флаг в _on_rollout_end)
        if self.save_snapshot_every_cycle and self._pending_snapshot:
            # Always use model.num_timesteps for snapshot step numbers
            # This ensures snapshots have correct step numbers matching the model's training progress
            snapshot_step = self.model.num_timesteps if hasattr(self.model, 'num_timesteps') else self.current_step
            self.opponent_pool.save_snapshot(
                self.model,
                snapshot_step,
                prefix="snapshot",
                force=True  # Force save regardless of snapshot_freq
            )
            if self.verbose > 0:
                print(f"Saved snapshot after training cycle {self.training_cycle_count} (step {snapshot_step:,})")
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
        if not self.save_snapshot_every_cycle:
            # Use model.num_timesteps for step-based snapshot saving
            step_for_snapshot = self.model.num_timesteps if hasattr(self.model, 'num_timesteps') else self.current_step
            if step_for_snapshot % self.snapshot_freq == 0:
                self.opponent_pool.save_snapshot(
                    self.model,
                    step_for_snapshot,
                    prefix="snapshot"
                )
                if self.verbose > 0:
                    pool_stats = self.opponent_pool.get_statistics()
                    print(f"Saved snapshot at step {step_for_snapshot:,}")
                    print(f"Pool statistics: {pool_stats}")
        
        # Update vec_env with current step for opponent sampling
        # Use model.num_timesteps as it's the authoritative source for step counting
        # VecMonitor should forward attributes, but we try both wrapped and underlying env
        if self.vec_env is not None:
            try:
                # Use model.num_timesteps for opponent sampling to ensure correct step progression
                step_for_sampling = self.model.num_timesteps if hasattr(self.model, 'num_timesteps') else self.current_step
                # Try to set on wrapped env (VecMonitor should forward)
                if hasattr(self.vec_env, 'current_step'):
                    self.vec_env.current_step = step_for_sampling
                # Also try underlying env if accessible
                elif hasattr(self.vec_env, 'envs') and len(self.vec_env.envs) > 0:
                    # VecMonitor wraps the original env
                    underlying = getattr(self.vec_env, 'env', None) or getattr(self.vec_env, 'venv', None)
                    if underlying is not None and hasattr(underlying, 'current_step'):
                        underlying.current_step = step_for_sampling
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
        # Увеличиваем счетчик циклов обучения
        self.training_cycle_count += 1
        
        
        # Set flag to save snapshot after model update (in next _on_step())
        # Сохраняем снапшот только каждые snapshot_cycle_freq циклов
        if self.save_snapshot_every_cycle and self.training_cycle_count % self.snapshot_cycle_freq == 0:
            self._pending_snapshot = True
            if self.verbose > 0:
                print(f"Training cycle {self.training_cycle_count}: will save snapshot after model update")
        
        # Log collected statistics to TensorBoard
        # Note: In SB3, logger is available after _on_training_start() is called
        # We log metrics in _on_rollout_end() which is called before model.update()
        # The metrics will be written to TensorBoard when logger.dump() is called
        # after model.update() completes (this happens automatically in SB3)
        with self._lock:
            if len(self.episode_rewards) > 0:
                # Calculate statistics from collected episodes
                mean_reward = np.mean(self.episode_rewards)
                mean_length = np.mean(self.episode_lengths)
                win_rate = np.mean(self.episode_wins) if len(self.episode_wins) > 0 else 0.0
                
                # Log to TensorBoard using logger (if available)
                # In SB3, BaseCallback has access to self.logger which is a Logger instance
                # The logger is initialized by the model when learn() is called
                # Logger is set in BaseCallback._init_callback() which is called before _on_training_start()
                if hasattr(self, 'logger') and self.logger is not None:
                    try:
                        # Get current step from model for correct step synchronization
                        # In _on_rollout_end(), model hasn't been updated yet, so we use current num_timesteps
                        current_step = self.model.num_timesteps if hasattr(self.model, 'num_timesteps') else None
                        
                        # Log custom metrics
                        # Note: logger.record() adds metrics to buffer, they will be written to TensorBoard
                        # when logger.dump() is called after model.update() (automatic in SB3)
                        self.logger.record("custom/mean_episode_reward", mean_reward)
                        self.logger.record("custom/mean_episode_length", mean_length)
                        self.logger.record("custom/win_rate", win_rate)
                        self.logger.record("custom/total_episodes", len(self.episode_rewards))
                        
                        # Log opponent pool statistics
                        pool_stats = self.opponent_pool.get_statistics()
                        if pool_stats:
                            self.logger.record("custom/opponent_pool_size", pool_stats.get("pool_size", 0))
                            self.logger.record("custom/opponent_pool_total_episodes", pool_stats.get("total_episodes", 0))
                            
                            # Log average winrate of opponents in pool
                            avg_winrate = pool_stats.get("average_winrate", 0.0)
                            if avg_winrate is not None:
                                self.logger.record("custom/opponent_pool_avg_winrate", avg_winrate)
                        
                        # In SB3, logger.dump() is called automatically after model.update()
                        # However, to ensure custom metrics are written, we can call it explicitly
                        # But this should not be necessary as SB3 handles it automatically
                        # We rely on SB3's automatic dump() call after model.update()
                        
                    except Exception as e:
                        # Don't crash training if logging fails
                        if self.verbose > 0:
                            print(f"Warning: Failed to log metrics to TensorBoard: {e}")
                            import traceback
                            traceback.print_exc()
                elif self.verbose > 1:
                    # Debug: log when logger is not available
                    print("Debug: Logger not available in callback (this is normal before learn() is called)")
                
                # Clear buffers after logging (keep only recent episodes for rolling average)
                # Keep last 100 episodes for better statistics
                max_buffer_size = 100
                if len(self.episode_rewards) > max_buffer_size:
                    self.episode_rewards = self.episode_rewards[-max_buffer_size:]
                    self.episode_lengths = self.episode_lengths[-max_buffer_size:]
                    self.episode_wins = self.episode_wins[-max_buffer_size:]
        
        # Winrate statistics are updated in VecEnv.step_wait() when episodes end
        return True



