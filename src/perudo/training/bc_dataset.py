"""
Dataset class for Behavioral Cloning training on collected trajectories.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


def convert_observation_to_tensor(obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    """
    Convert observation dictionary from numpy arrays to torch tensors.
    
    Args:
        obs: Observation dictionary with keys 'bid_history', 'static_info', 'action_mask'
        
    Returns:
        Dictionary with same keys but torch.Tensor values
    """
    tensor_obs = {}
    
    # Convert bid_history: (max_history_length, 2) int32 -> torch.int32
    if "bid_history" in obs:
        bid_history = obs["bid_history"]
        if isinstance(bid_history, np.ndarray):
            tensor_obs["bid_history"] = torch.from_numpy(bid_history).to(torch.int32)
        else:
            tensor_obs["bid_history"] = torch.tensor(bid_history, dtype=torch.int32)
    
    # Convert static_info: (static_info_size,) float32 -> torch.float32
    if "static_info" in obs:
        static_info = obs["static_info"]
        if isinstance(static_info, np.ndarray):
            tensor_obs["static_info"] = torch.from_numpy(static_info).to(torch.float32)
        else:
            tensor_obs["static_info"] = torch.tensor(static_info, dtype=torch.float32)
    
    # Convert action_mask: (action_space.n,) bool -> torch.bool
    if "action_mask" in obs:
        action_mask = obs["action_mask"]
        if isinstance(action_mask, np.ndarray):
            tensor_obs["action_mask"] = torch.from_numpy(action_mask).to(torch.bool)
        else:
            tensor_obs["action_mask"] = torch.tensor(action_mask, dtype=torch.bool)
    
    return tensor_obs


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset for behavioral cloning training on collected trajectories.
    
    Handles conversion of observations (Dict) to tensors and manages action masks.
    """
    
    def __init__(
        self,
        observations: List[Dict[str, np.ndarray]],
        actions: List[int],
        action_masks: Optional[List[Optional[np.ndarray]]] = None,
    ):
        """
        Initialize trajectory dataset.
        
        Args:
            observations: List of observation dictionaries
            actions: List of action integers
            action_masks: Optional list of action mask arrays (can contain None values)
        """
        self.observations = observations
        self.actions = actions
        self.action_masks = action_masks if action_masks is not None else [None] * len(observations)
        
        # Validate lengths
        if len(self.observations) != len(self.actions):
            raise ValueError(f"Observations ({len(self.observations)}) and actions ({len(self.actions)}) must have same length")
        
        if len(self.action_masks) != len(self.observations):
            raise ValueError(f"Action masks ({len(self.action_masks)}) must have same length as observations ({len(self.observations)})")
        
        # Convert actions to numpy array for easier handling
        self.actions = np.array(self.actions, dtype=np.int64)
        
        logger.info(f"Created TrajectoryDataset with {len(self)} samples")
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.observations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with keys:
            - 'observation': Dict[str, torch.Tensor] - converted observation
            - 'action': torch.Tensor - action as tensor
            - 'action_mask': Optional[torch.Tensor] - action mask as tensor (or None)
        """
        obs = self.observations[idx]
        action = self.actions[idx]
        action_mask = self.action_masks[idx]
        
        # Convert observation to tensors
        obs_tensor = convert_observation_to_tensor(obs)
        
        # Convert action to tensor
        action_tensor = torch.tensor(action, dtype=torch.long)
        
        # Convert action mask to tensor if present
        action_mask_tensor = None
        if action_mask is not None:
            if isinstance(action_mask, np.ndarray):
                action_mask_tensor = torch.from_numpy(action_mask).to(torch.bool)
            else:
                action_mask_tensor = torch.tensor(action_mask, dtype=torch.bool)
        
        return {
            "observation": obs_tensor,
            "action": action_tensor,
            "action_mask": action_mask_tensor,
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for DataLoader to handle Dict observations.
    
    Args:
        batch: List of samples from TrajectoryDataset
        
    Returns:
        Batched dictionary with:
        - 'observation': Dict[str, torch.Tensor] - batched observations
        - 'action': torch.Tensor - batched actions
        - 'action_mask': Optional[torch.Tensor] - batched action masks
    """
    # Extract all observations, actions, and masks
    observations = [item["observation"] for item in batch]
    actions = [item["action"] for item in batch]
    action_masks = [item["action_mask"] for item in batch]
    
    # Batch observations (Dict of tensors)
    # Each observation is a Dict with keys like 'bid_history', 'static_info', 'action_mask'
    batched_obs = {}
    if observations:
        # Get keys from first observation
        keys = observations[0].keys()
        for key in keys:
            # Stack tensors along batch dimension
            tensors = [obs[key] for obs in observations]
            
            # Special handling for bid_history which may have variable length
            if key == "bid_history":
                # Find maximum history length in batch
                max_length = max(t.shape[0] for t in tensors)
                
                # Pad all tensors to max_length
                padded_tensors = []
                for t in tensors:
                    if t.shape[0] < max_length:
                        # Pad with zeros (padding value is [0, 0] for bid_history)
                        padding = torch.zeros(
                            max_length - t.shape[0], 
                            t.shape[1], 
                            dtype=t.dtype, 
                            device=t.device
                        )
                        t_padded = torch.cat([t, padding], dim=0)
                    else:
                        t_padded = t
                    padded_tensors.append(t_padded)
                
                batched_obs[key] = torch.stack(padded_tensors, dim=0)
            else:
                # For other keys, stack directly (they should have same shape)
                batched_obs[key] = torch.stack(tensors, dim=0)
    
    # Batch actions
    batched_actions = torch.stack(actions, dim=0)
    
    # Batch action masks (handle None values)
    batched_action_masks = None
    if any(mask is not None for mask in action_masks):
        # If any mask is not None, create batched mask
        # For None masks, create a mask of all True (all actions valid)
        mask_tensors = []
        action_space_size = None
        
        # First, find action space size from a non-None mask
        for mask in action_masks:
            if mask is not None:
                if isinstance(mask, torch.Tensor):
                    action_space_size = mask.shape[0] if mask.dim() == 1 else mask.shape[-1]
                else:
                    action_space_size = len(mask) if hasattr(mask, '__len__') else None
                break
        
        # If we couldn't determine size, try to infer from observation
        if action_space_size is None and observations:
            # Try to get action space size from observation's action_mask if present
            first_obs = observations[0]
            if "action_mask" in first_obs:
                obs_mask = first_obs["action_mask"]
                if isinstance(obs_mask, torch.Tensor):
                    action_space_size = obs_mask.shape[0] if obs_mask.dim() == 1 else obs_mask.shape[-1]
                elif isinstance(obs_mask, np.ndarray):
                    action_space_size = obs_mask.shape[0] if obs_mask.ndim == 1 else obs_mask.shape[-1]
        
        # Now create masks
        for mask in action_masks:
            if mask is not None:
                mask_tensors.append(mask)
            else:
                # Create a mask of all True (all actions valid)
                if action_space_size is not None:
                    all_true_mask = torch.ones(action_space_size, dtype=torch.bool)
                else:
                    # Fallback: use shape from first valid mask if available
                    if mask_tensors:
                        all_true_mask = torch.ones_like(mask_tensors[0], dtype=torch.bool)
                    else:
                        # Cannot determine size, skip this batch's mask
                        continue
                mask_tensors.append(all_true_mask)
        
        if mask_tensors:
            batched_action_masks = torch.stack(mask_tensors, dim=0)
    else:
        batched_action_masks = None
    
    return {
        "observation": batched_obs,
        "action": batched_actions,
        "action_mask": batched_action_masks,
    }

