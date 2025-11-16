"""
Utility functions for training Perudo agents.

Contains helper functions for:
- Device detection (GPU/CPU)
- Learning rate scheduling
- Model file management
"""

import os
import re
import glob
import shutil
import json
from typing import Optional
import torch


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



