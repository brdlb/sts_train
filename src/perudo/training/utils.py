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
import logging
from typing import Optional, Callable, List
import torch

# Setup logger for this module
logger = logging.getLogger(__name__)


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
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("CUDA not available, using CPU")
    
    return device


def linear_schedule(initial_value: float, final_value: Optional[float] = None, decay_ratio: float = 0.3) -> Callable[[float], float]:
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


def find_latest_model(model_dir: str, additional_dirs: Optional[List[str]] = None) -> Optional[str]:
    """
    Find the latest saved model in the model directory and optional additional directories.
    
    Looks for any .zip files and returns the most recent one by modification time.
    
    Args:
        model_dir: Primary directory to search for models
        additional_dirs: Optional list of additional directories to search
    
    Returns:
        Path to the latest model (by modification time), or None if no models found
    """
    search_dirs = [model_dir]
    if additional_dirs:
        search_dirs.extend(additional_dirs)

    print(search_dirs)
    
    all_zip_files = []
    
    for d in search_dirs:
        if not os.path.exists(d):
            continue
            
        # Find all .zip files in the directory
        zip_pattern = os.path.join(d, "*.zip")
        zip_files = glob.glob(zip_pattern)
        all_zip_files.extend(zip_files)
    
    if not all_zip_files:
        return None
    
    # Sort by modification time (most recent first)
    all_zip_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Return the most recent file
    print(all_zip_files[0])
    return all_zip_files[0]


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
        logger.warning(f"Could not load pool metadata: {e}")
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
        logger.info(f"Restored model from opponent pool: {best_snapshot['id']}")
        logger.info(f"  Source: {source_path}")
        logger.info(f"  Target: {target_path}")
        logger.info(f"  Step: {best_snapshot['step']}, ELO: {best_snapshot['elo']:.2f}")
        return target_path
    except Exception as e:
        logger.warning(f"Failed to copy model from opponent pool: {e}")
        return None



