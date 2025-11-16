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

from ..game.perudo_vec_env import PerudoMultiAgentVecEnv
from .config import Config, DEFAULT_CONFIG
from .opponent_pool import OpponentPool
from .bot_personality_tracker import BotPersonalityTracker
from ..agents.transformer_extractor import CustomTransformerExtractor

import sys
import torch

# Global debug mode flag (thread-safe)
_debug_mode = threading.Event()
_debug_mode_lock = threading.Lock()

def get_debug_mode() -> bool:
    """Get current debug mode state (thread-safe)."""
    with _debug_mode_lock:
        return _debug_mode.is_set()

def set_debug_mode(enabled: bool) -> None:
    """Set debug mode state (thread-safe)."""
    with _debug_mode_lock:
        if enabled:
            _debug_mode.set()
            print("\n[DEBUG MODE] ON - Выводится каждый ход в игре")
        else:
            _debug_mode.clear()
            print("\n[DEBUG MODE] OFF - Обучение продолжается в обычном режиме")

def _keyboard_listener_thread():
    """Thread function to listen for keyboard shortcuts (Ctrl+D to toggle debug mode)."""
    try:
        from pynput import keyboard
        
        # Track last toggle time to prevent rapid toggling
        last_toggle_time = [0.0]
        toggle_lock = threading.Lock()
        
        def toggle_debug_mode():
            """Toggle debug mode when Ctrl+D is pressed."""
            try:
                import time
                current_time = time.time()
                
                # Prevent rapid toggling (minimum 0.3 seconds between toggles)
                with toggle_lock:
                    if current_time - last_toggle_time[0] < 0.3:
                        return
                    last_toggle_time[0] = current_time
                
                current_state = get_debug_mode()
                set_debug_mode(not current_state)
            except Exception as e:
                # Silently ignore errors to prevent crashes
                pass
        
        # Use GlobalHotKeys for better cross-platform support
        # This is the recommended way to handle global hotkeys
        hotkey_string = '<ctrl>+d'
        
        # Create GlobalHotKeys listener
        def on_activate():
            """Called when hotkey is pressed."""
            toggle_debug_mode()
        
        # Try using GlobalHotKeys first (more reliable)
        try:
            print("[DEBUG MODE] Keyboard listener started. Press Ctrl+D to toggle debug mode.")
            with keyboard.GlobalHotKeys({hotkey_string: on_activate}) as listener:
                listener.join()
        except Exception as e:
            # If GlobalHotKeys fails, try manual tracking
            print(f"[DEBUG MODE] GlobalHotKeys failed, using fallback method. Error: {e}")
            # Fallback to manual tracking if GlobalHotKeys doesn't work
            pressed_keys = set()
            pressed_keys_lock = threading.Lock()
            
            def on_press(key):
                """Handle key press events."""
                try:
                    with pressed_keys_lock:
                        # Track Ctrl key (both left and right)
                        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r, keyboard.Key.ctrl):
                            pressed_keys.add('ctrl')
                        # Track 'd' key
                        elif hasattr(key, 'char') and key.char and key.char.lower() == 'd':
                            pressed_keys.add('d')
                        elif hasattr(key, 'name') and key.name and key.name.lower() == 'd':
                            pressed_keys.add('d')
                        
                        # Check if Ctrl+D is pressed
                        if 'ctrl' in pressed_keys and 'd' in pressed_keys:
                            toggle_debug_mode()
                            # Clear to prevent multiple toggles
                            pressed_keys.clear()
                except Exception:
                    pass
            
            def on_release(key):
                """Handle key release events."""
                try:
                    with pressed_keys_lock:
                        # Remove released keys from tracking
                        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r, keyboard.Key.ctrl):
                            pressed_keys.discard('ctrl')
                        elif hasattr(key, 'char') and key.char and key.char.lower() == 'd':
                            pressed_keys.discard('d')
                        elif hasattr(key, 'name') and key.name and key.name.lower() == 'd':
                            pressed_keys.discard('d')
                except Exception:
                    pass
            
            # Start listening to keyboard events
            print("[DEBUG MODE] Fallback keyboard listener started. Press Ctrl+D to toggle debug mode.")
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                listener.join()
    
    except ImportError:
        # If pynput is not available, use fallback method with stdin
        print("Warning: pynput not available. Using stdin-based debug mode toggle.")
        print("Press 'd' + Enter to toggle debug mode, or Ctrl+C to exit.")
        
        import sys
        import time
        
        # Use a separate thread for stdin reading to avoid blocking
        def stdin_reader():
            """Read stdin in a separate thread."""
            while True:
                try:
                    line = sys.stdin.readline().strip().lower()
                    if line == 'd':
                        current_state = get_debug_mode()
                        set_debug_mode(not current_state)
                except (KeyboardInterrupt, EOFError):
                    break
                except Exception:
                    # Silently ignore errors to prevent crashes
                    time.sleep(0.1)
        
        # Start stdin reader thread
        stdin_thread = threading.Thread(target=stdin_reader, daemon=True)
        stdin_thread.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
    except Exception as e:
        # Silently ignore errors to prevent crashes
        pass

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


def linear_schedule(initial_value: float, final_value: Optional[float] = None, decay_ratio: float = 0.3):
    """
    Linear learning rate schedule with slower decay.
    
    Args:
        initial_value: Initial learning rate
        final_value: Final learning rate. If None, uses decay_ratio * initial_value
        decay_ratio: Ratio of final to initial learning rate (default: 0.3, meaning final is 30% of initial)
    
    Returns:
        Callable that takes progress_remaining (1.0 to 0.0) and returns learning rate
    """
    if final_value is None:
        final_value = initial_value * decay_ratio
    
    def func(progress_remaining: float) -> float:
        """
        Calculate learning rate based on training progress.
        
        Args:
            progress_remaining: Progress from 1.0 (start) to 0.0 (end)
        
        Returns:
            Current learning rate
        """
        return final_value + (initial_value - final_value) * progress_remaining
    
    return func


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


def restore_model_from_opponent_pool(model_dir: str, pool_dir: str) -> Optional[str]:
    """
    Restore model from opponent pool if no model exists in the root directory.
    
    This function checks if there are any models in the root model directory.
    If not, it looks for the best model in the opponent pool (by step count or ELO)
    and copies it to the root directory with the correct naming format.
    
    Args:
        model_dir: Root directory for models (e.g., "models")
        pool_dir: Directory containing opponent pool (e.g., "models/opponent_pool")
    
    Returns:
        Path to the restored model in root directory, or None if no model was found in pool
    """
    # Check if there's already a model in root
    existing_model = find_latest_model(model_dir)
    if existing_model and os.path.exists(existing_model):
        return None  # Model already exists, no need to restore
    
    # Check if pool directory exists
    if not os.path.exists(pool_dir):
        return None
    
    # Load pool metadata
    metadata_file = os.path.join(pool_dir, "pool_metadata.json")
    if not os.path.exists(metadata_file):
        return None
    
    try:
        with open(metadata_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load pool metadata: {e}")
        return None
    
    snapshots = data.get("snapshots", {})
    if not snapshots:
        return None
    
    # Find the best snapshot (by step count - prefer the most recent training step)
    best_snapshot = None
    best_step = -1
    
    for snapshot_id, snapshot_data in snapshots.items():
        snapshot_path = snapshot_data.get("path", "")
        step = snapshot_data.get("step", 0)
        
        # Handle both absolute and relative paths
        # If path is absolute but doesn't exist, try to find by filename in pool_dir
        if os.path.isabs(snapshot_path):
            if not os.path.exists(snapshot_path):
                # Try to find file by name in pool_dir
                filename = os.path.basename(snapshot_path)
                fallback_path = os.path.join(pool_dir, filename)
                if os.path.exists(fallback_path):
                    snapshot_path = fallback_path
                else:
                    continue  # Skip if file doesn't exist
        else:
            # Relative path - construct absolute path
            snapshot_path = os.path.join(pool_dir, os.path.basename(snapshot_path))
        
        # Check if file exists
        if not os.path.exists(snapshot_path):
            continue
        
        # Prefer snapshot with highest step count
        if step > best_step:
            best_step = step
            best_snapshot = {
                "id": snapshot_id,
                "path": snapshot_path,
                "step": step,
                "elo": snapshot_data.get("elo", 1500.0),
            }
    
    if best_snapshot is None:
        return None
    
    # Copy the best snapshot to root directory with correct naming
    source_path = best_snapshot["path"]
    target_filename = f"perudo_model_{best_snapshot['step']}_steps.zip"
    target_path = os.path.join(model_dir, target_filename)
    
    try:
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Copy file
        shutil.copy2(source_path, target_path)
        print(f"Restored model from opponent pool: {best_snapshot['id']}")
        print(f"  Source: {source_path}")
        print(f"  Target: {target_path}")
        print(f"  Step: {best_snapshot['step']}, ELO: {best_snapshot['elo']:.2f}")
        return target_path
    except Exception as e:
        print(f"Warning: Failed to copy model from opponent pool: {e}")
        return None


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
        # The advantages will be normalized in PPO.update() by default
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
        
        # Log training cycle completion
        current_step = self.model.num_timesteps if hasattr(self.model, 'num_timesteps') else self.current_step
        n_steps = getattr(self.model, 'n_steps', 'N/A')
        if self.verbose > 0:
            print(f"\n{'='*70}")
            print(f"✓ Training cycle {self.training_cycle_count} completed!")
            print(f"  Total steps: {current_step}")
            print(f"  Steps per cycle: {n_steps}")
            print(f"  Next cycle: collecting {n_steps} more steps before next update...")
            print(f"{'='*70}\n")
        
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
        
        # Create directories
        # Ensure paths are absolute for consistency
        log_dir = os.path.abspath(config.training.log_dir)
        model_dir = os.path.abspath(config.training.model_dir)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Update config with absolute paths
        config.training.log_dir = log_dir
        config.training.model_dir = model_dir
        
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
        
        # Create bot personality tracker for tracking bot statistics
        bot_stats_file = os.path.join(config.training.model_dir, "bot_personality_stats.json")
        self.bot_personality_tracker = BotPersonalityTracker(
            stats_file=bot_stats_file,
            elo_k=32,
        )
        
        # Create vectorized environment
        # Use transformer_history_length from config if available, otherwise use history_length
        max_history_length = getattr(config.training, 'transformer_history_length', config.game.history_length)
        use_bot_opponents = getattr(config.training, 'use_bot_opponents', False)
        bot_personalities = getattr(config.training, 'bot_personalities', None)
        
        vec_env_raw = PerudoMultiAgentVecEnv(
            num_envs=self.num_envs,
            num_players=self.num_players,
            dice_per_player=config.game.dice_per_player,
            total_dice_values=config.game.total_dice_values,
            max_quantity=config.game.max_quantity,
            history_length=config.game.history_length,
            max_history_length=max_history_length,
            opponent_pool=self.opponent_pool if not use_bot_opponents else None,  # Don't use pool if using bots
            random_num_players=config.game.random_num_players,
            min_players=config.game.min_players,
            max_players=config.game.max_players,
            reward_config=config.reward,
            use_bot_opponents=use_bot_opponents,
            bot_personalities=bot_personalities,
            bot_personality_tracker=self.bot_personality_tracker if use_bot_opponents else None,
        )
        
        # Wrap with ActionMasker to enable action masking for MaskablePPO
        # ActionMasker extracts action_mask from observation dict and provides action_masks() method
        def mask_fn(env):
            return env.reset()[0]["action_mask"].astype(bool)
        
        # For VecEnv, we need to wrap each individual environment
        # ActionMasker doesn't work directly with VecEnv, so we need to wrap individual envs
        # However, PerudoMultiAgentVecEnv returns Dict observations, so we need a custom approach
        # Instead, we'll modify the VecEnv to support action masking directly
        # For now, let's try wrapping the VecEnv directly (ActionMasker might support VecEnv)
        
        # Actually, ActionMasker is designed for single environments, not VecEnv
        # We need to use a different approach - create a wrapper that extracts masks from observations
        # Or use a VecActionMasker if it exists, or create our own
        
        # Wrap with VecMonitor for logging
        monitor_dir = os.path.join(config.training.log_dir, "monitor")
        os.makedirs(monitor_dir, exist_ok=True)
        vec_env_monitored = VecMonitor(vec_env_raw, monitor_dir)
        
        # Explicitly forward action_masks() method from underlying VecEnv
        # VecMonitor doesn't automatically forward custom methods like action_masks
        def action_masks(env_self):
            """Forward action_masks from underlying VecEnv."""
            # VecMonitor stores underlying VecEnv in .venv attribute
            underlying = getattr(env_self, 'venv', None)
            if underlying is not None and hasattr(underlying, 'action_masks'):
                return underlying.action_masks()
            # Fallback: try direct access to vec_env_raw (stored as reference)
            if hasattr(env_self, '_vec_env_raw') and hasattr(env_self._vec_env_raw, 'action_masks'):
                return env_self._vec_env_raw.action_masks()
            # Final fallback: return all actions as valid
            return np.ones((env_self.num_envs, env_self.action_space.n), dtype=bool)
        
        # Override has_attr to check for action_masks in both VecMonitor and underlying VecEnv
        # This is needed because is_masking_supported() uses has_attr() to check for action_masks
        original_has_attr = vec_env_monitored.has_attr
        def has_attr(env_self, attr_name: str) -> bool:
            """Check if attribute exists in VecMonitor or underlying VecEnv."""
            # First check VecMonitor itself (for dynamically added methods like action_masks)
            if hasattr(env_self, attr_name):
                return True
            # Then check underlying VecEnv using original has_attr
            return original_has_attr(attr_name)
        
        import types
        vec_env_monitored.action_masks = types.MethodType(action_masks, vec_env_monitored)
        vec_env_monitored.has_attr = types.MethodType(has_attr, vec_env_monitored)
        # Store reference to underlying env for action_masks method
        vec_env_monitored._vec_env_raw = vec_env_raw
        
        self.vec_env = vec_env_monitored
        
        # Store reference to original env for direct access if needed
        # VecMonitor should forward attributes, but we keep reference just in case
        self._vec_env_raw = vec_env_raw
        
        # Determine device for training (GPU with CPU fallback)
        device = get_device(config.training.device)
        
        # Try to restore model from opponent pool if no model exists in root
        # This allows continuing training from opponent pool snapshots
        restored_model_path = restore_model_from_opponent_pool(
            config.training.model_dir,
            pool_dir
        )
        if restored_model_path:
            print(f"Successfully restored model from opponent pool to: {restored_model_path}")
        
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
                # Ensure tensorboard_log is set to absolute path
                self.model.tensorboard_log = os.path.abspath(config.training.log_dir)
                
                # Update learning rate schedule for continued training
                # Use slower decay: final LR is 50% of initial (slower decay for better learning)
                initial_lr = config.training.learning_rate
                lr_schedule = linear_schedule(initial_lr, decay_ratio=0.5)
                self.model.learning_rate = lr_schedule
                final_lr = initial_lr * 0.5
                
                print(f"Successfully loaded model from {latest_model_path}")
                print(f"TensorBoard logging enabled: {self.model.tensorboard_log}")
                print(f"Learning rate schedule updated: {initial_lr:.2e} -> {final_lr:.2e} (linear decay, 50% of initial)")
                
                # Get current timesteps from model (SB3 saves this in the model)
                self.initial_timesteps = self.model.num_timesteps if hasattr(self.model, 'num_timesteps') else 0
                
                # Extract step count from filename if possible (for verification)
                match = re.search(r"perudo_model_(\d+)_steps\.zip", os.path.basename(latest_model_path))
                if match:
                    filename_steps = int(match.group(1))
                    print(f"Model filename indicates {filename_steps} steps. Model num_timesteps: {self.initial_timesteps}")
                    
                    # If model.num_timesteps doesn't match filename, use filename value
                    # This can happen if model was saved but num_timesteps wasn't properly saved
                    if self.initial_timesteps == 0 or abs(self.initial_timesteps - filename_steps) > 1000:
                        print(f"Warning: Model num_timesteps ({self.initial_timesteps}) doesn't match filename ({filename_steps})")
                        print(f"Using filename steps ({filename_steps}) as initial_timesteps")
                        self.initial_timesteps = filename_steps
                        # Try to set num_timesteps in model if possible (SB3 might not allow this directly)
                        # We'll rely on the corrected total_timesteps calculation instead
                else:
                    print(f"Continuing training from {self.initial_timesteps} steps.")
                
                if self.initial_timesteps > 0:
                    print(f"Training will continue from step {self.initial_timesteps:,}")
                else:
                    print(f"Warning: initial_timesteps is 0. Training will start from the beginning.")
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
                    features_extractor_class=CustomTransformerExtractor,
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
                    # Enhanced network architectures for policy and value networks
                    # Value network receives 256-dim features, needs stronger architecture
                    # Format: list with dict specifying separate architectures for pi and vf
                    net_arch=[dict(pi=[192, 128], vf=[256, 128])],
                )
            else:
                policy_kwargs = config.training.policy_kwargs
            
            # Create learning rate schedule: linear decay with slower rate
            # Final LR is 50% of initial (slower decay for better learning throughout training)
            initial_lr = config.training.learning_rate
            lr_schedule = linear_schedule(initial_lr, decay_ratio=0.5)
            final_lr = initial_lr * 0.5
            
            self.model = MaskablePPO(
                policy=config.training.policy,
                env=self.vec_env,
                device=device,
                policy_kwargs=policy_kwargs,
                learning_rate=lr_schedule,  # Use learning rate schedule
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
                tensorboard_log=os.path.abspath(config.training.log_dir),  # Enable TensorBoard logging (absolute path)
            )
            
            print(f"TensorBoard logging enabled: {self.model.tensorboard_log}")
            print(f"Learning rate schedule: {initial_lr:.2e} -> {final_lr:.2e} (linear decay, 50% of initial)")
            
            # For new models, initial_timesteps is 0
            self.initial_timesteps = 0
            print(f"Starting new training from step 0")
        
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
        
        # Start keyboard listener thread for debug mode toggle
        keyboard_thread = threading.Thread(target=_keyboard_listener_thread, daemon=True)
        keyboard_thread.start()
        
        # Enable debug mode by default
        set_debug_mode(False)

        
        # Store debug mode flag in vec_env for access during training
        self._vec_env_raw.debug_mode = _debug_mode
        
        # Adjust total_timesteps to account for already completed training
        # If model was loaded with initial_timesteps > 0, we need to train for
        # initial_timesteps + config.training.total_timesteps to get the correct
        # progress_remaining calculation for learning rate schedule
        config_total_timesteps = self.config.training.total_timesteps
        if self.initial_timesteps > 0:
            # Calculate total timesteps: train for additional config.training.total_timesteps steps
            # This ensures progress_remaining is calculated correctly
            total_timesteps = self.initial_timesteps + config_total_timesteps
            print(f"Continuing training:")
            print(f"  Initial timesteps: {self.initial_timesteps:,}")
            print(f"  Additional timesteps: {config_total_timesteps:,}")
            print(f"  Total timesteps: {total_timesteps:,}")
            print(f"  Target progress: {self.initial_timesteps / total_timesteps * 100:.1f}% -> 100%")
        else:
            # New training: use config total_timesteps directly
            total_timesteps = config_total_timesteps
            print(f"Starting new training:")
            print(f"  Total timesteps: {total_timesteps:,}")
        
        # Calculate effective buffer size
        # Note: In multi-agent setup, only agent_id=0 is learning, but we collect data from all agents
        # The actual learning buffer size is n_steps * num_envs (one step per env per collection step)
        effective_buffer_size = self.config.training.n_steps * self.num_envs
        steps_per_env = self.config.training.n_steps
        num_batches_per_cycle = effective_buffer_size // self.config.training.batch_size
        gradient_updates_per_cycle = num_batches_per_cycle * self.config.training.n_epochs
        
        print(f"Training configuration:")
        print(f"  n_steps: {self.config.training.n_steps} (steps to collect before update)")
        print(f"  batch_size: {self.config.training.batch_size} (mini-batch size)")
        print(f"  n_epochs: {self.config.training.n_epochs} (epochs per update cycle)")
        print(f"  Effective buffer size: {effective_buffer_size:,} samples")
        print(f"  Steps per environment: {steps_per_env} (~{steps_per_env/1000:.1f}K steps/env)")
        print(f"  Batches per cycle: {num_batches_per_cycle}")
        print(f"  Gradient updates per cycle: {gradient_updates_per_cycle:,}")
        
        # Check that batch_size divides evenly
        if effective_buffer_size % self.config.training.batch_size != 0:
            print(f"Warning: Effective buffer size {effective_buffer_size} does not divide evenly "
                  f"by batch_size {self.config.training.batch_size}")
            print(f"Consider adjusting batch_size (current: {self.config.training.batch_size})")
            print(f"  Suggested batch_size values that divide evenly: ", end="")
            # Suggest divisors
            suggested_sizes = [s for s in [128, 256, 512, 1024] if effective_buffer_size % s == 0]
            print(", ".join(map(str, suggested_sizes)) if suggested_sizes else "none found")
        
        # Create callbacks
        callbacks = []
        
        # Advantage normalization callback
        adv_norm_callback = AdvantageNormalizationCallback(verbose=self.config.training.verbose)
        callbacks.append(adv_norm_callback)
        
        # Adaptive entropy callback (if enabled)
        if getattr(self.config.training, 'adaptive_entropy', False):
            adaptive_entropy_callback = AdaptiveEntropyCallback(
                threshold_low=self.config.training.entropy_threshold_low,
                threshold_high=self.config.training.entropy_threshold_high,
                adjustment_rate=self.config.training.entropy_adjustment_rate,
                max_ent_coef=self.config.training.entropy_max_coef,
                verbose=self.config.training.verbose,
            )
            callbacks.append(adaptive_entropy_callback)
            print(f"Adaptive entropy callback enabled:")
            print(f"  Thresholds: low={adaptive_entropy_callback.threshold_low:.2f}, high={adaptive_entropy_callback.threshold_high:.2f}")
            print(f"  Adjustment rate: {adaptive_entropy_callback.adjustment_rate:.4f}")
            print(f"  Max ent_coef: {adaptive_entropy_callback.max_ent_coef:.4f}")
        
        # Self-play callback
        selfplay_callback = SelfPlayTrainingCallback(
            opponent_pool=self.opponent_pool,
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
        
        # Verify that model.num_timesteps matches initial_timesteps
        # If not, we might need to manually set it (though SB3 should handle this)
        current_model_timesteps = self.model.num_timesteps if hasattr(self.model, 'num_timesteps') else 0
        if self.initial_timesteps > 0 and current_model_timesteps != self.initial_timesteps:
            print(f"Warning: Model num_timesteps ({current_model_timesteps}) doesn't match initial_timesteps ({self.initial_timesteps})")
            print(f"SB3 should handle this automatically, but progress calculations might be affected.")
            # Note: We can't directly set model.num_timesteps in SB3, but the learn() method
            # should respect the current value. The total_timesteps adjustment above should
            # ensure progress_remaining is calculated correctly.
        
        self.model.learn(
            total_timesteps=total_timesteps,
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
        
        # Print bot personality statistics if using bot opponents
        if self.bot_personality_tracker is not None:
            self.bot_personality_tracker.print_summary()
        
    def save_model(self, path: str):
        """Save the current model."""
        self.model.save(path)
        
    def load_model(self, path: str):
        """Load a model."""
        # Suppress SB3 wrapping messages
        with contextlib.redirect_stdout(StringIO()):
            self.model = MaskablePPO.load(path, env=self.vec_env)
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
        "--n-steps",
        type=int,
        default=8192,
        help="Number of steps to collect before update (recommended: 8192 for 16 envs)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training (recommended: 256 for n_steps=8192)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training (cpu, cuda, cuda:0, etc.). If not specified, auto-detects GPU with CPU fallback.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level (0 = no output, 1 = progress bars and logs, 2 = debug). Default: 1",
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = DEFAULT_CONFIG
    # total_timesteps is set only in config.py, not overridden here
    config.training.n_steps = args.n_steps
    config.training.batch_size = args.batch_size
    config.training.device = args.device
    config.training.verbose = args.verbose  # Set verbose from command line
    
    # Override num_envs from command line if provided
    if args.num_envs is not None:
        config.training.num_envs = args.num_envs
    
    # Start training
    trainer = SelfPlayTraining(config)
    trainer.train()


if __name__ == "__main__":
    main()
