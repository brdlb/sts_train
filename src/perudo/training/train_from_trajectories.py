"""
Train agent from collected winner trajectories using imitation learning.

This script loads trajectories collected in botplay mode and trains an agent
using behavioral cloning or other imitation learning methods.
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.distributions import MaskableDistribution
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

from ..game.perudo_vec_env import PerudoMultiAgentVecEnv
from .train import ActionMaskVecMonitor
from ..agents.transformer_extractor import TransformerFeaturesExtractor
from .config import Config, DEFAULT_CONFIG
from .utils import get_device
from .bc_dataset import TrajectoryDataset, collate_batch

# Setup logger
logger = logging.getLogger(__name__)


def load_trajectories(data_dir: str) -> List[Dict]:
    """
    Load all collected trajectories from pickle files.
    
    Args:
        data_dir: Directory containing trajectory pickle files
        
    Returns:
        List of trajectory dictionaries
    """
    trajectories = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.warning(f"Trajectory directory does not exist: {data_dir}")
        return trajectories
    
    # Find all pickle files
    pickle_files = list(data_path.glob("trajectory_*.pkl"))
    
    if not pickle_files:
        logger.warning(f"No trajectory files found in {data_dir}")
        return trajectories
    
    logger.info(f"Loading {len(pickle_files)} trajectory files from {data_dir}")
    
    for pickle_file in pickle_files:
        try:
            with open(pickle_file, 'rb') as f:
                trajectory = pickle.load(f)
                trajectories.append(trajectory)
        except Exception as e:
            logger.warning(f"Failed to load trajectory from {pickle_file}: {e}")
    
    logger.info(f"Successfully loaded {len(trajectories)} trajectories")
    return trajectories


def prepare_training_data(
    trajectories: List[Dict],
    max_trajectories: Optional[int] = None
) -> Tuple[List[Dict], List[np.ndarray], List[np.ndarray], List[Optional[np.ndarray]]]:
    """
    Prepare training data from trajectories.
    
    Args:
        trajectories: List of trajectory dictionaries
        max_trajectories: Maximum number of trajectories to use (None = use all)
        
    Returns:
        Tuple of (observations, actions, rewards, action_masks)
        - observations: List of observation dictionaries
        - actions: List of action arrays
        - rewards: List of reward arrays (for reference, not used in BC)
        - action_masks: List of action mask arrays (optional)
    """
    if max_trajectories is not None:
        trajectories = trajectories[:max_trajectories]
    
    all_observations = []
    all_actions = []
    all_rewards = []
    all_action_masks = []
    
    total_steps = 0
    
    for traj in trajectories:
        observations = traj.get("observations", [])
        actions = traj.get("actions", [])
        rewards = traj.get("rewards", [])
        action_masks = traj.get("action_masks", [])
        
        # Ensure all lists have the same length
        min_length = min(len(observations), len(actions), len(rewards))
        
        all_observations.extend(observations[:min_length])
        all_actions.extend(actions[:min_length])
        all_rewards.extend(rewards[:min_length])
        
        # Action masks may be None or shorter
        if action_masks and len(action_masks) >= min_length:
            all_action_masks.extend(action_masks[:min_length])
        else:
            all_action_masks.extend([None] * min_length)
        
        total_steps += min_length
    
    logger.info(f"Prepared training data: {total_steps} steps from {len(trajectories)} trajectories")
    logger.info(f"  Average trajectory length: {total_steps / len(trajectories):.1f}")
    
    return all_observations, all_actions, all_rewards, all_action_masks


def create_model_from_config(
    config: Config,
    env: PerudoMultiAgentVecEnv,
    device: Optional[str] = None
) -> MaskablePPO:
    """
    Create MaskablePPO model from config (same as in train.py).
    
    Args:
        config: Training configuration
        env: Vectorized environment
        device: Device to use (None = auto-detect)
        
    Returns:
        Initialized MaskablePPO model
    """
    if device is None:
        device = get_device(config.training.device)
    
    # Build policy_kwargs with transformer extractor
    if config.training.policy_kwargs is None:
        obs_space = env.observation_space
        
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
            net_arch=[dict(pi=[96, 64], vf=[128, 64])],
        )
    else:
        policy_kwargs = config.training.policy_kwargs
    
    # Create learning rate schedule
    from .utils import linear_schedule
    initial_lr = config.training.learning_rate
    lr_schedule = linear_schedule(initial_lr, decay_ratio=0.5)
    
    model = MaskablePPO(
        policy=config.training.policy,
        env=env,
        device=device,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_schedule,
        n_steps=config.training.n_steps,
        batch_size=config.training.batch_size,
        n_epochs=config.training.n_epochs,
        gamma=config.training.gamma,
        gae_lambda=config.training.gae_lambda,
        clip_range=config.training.clip_range,
        ent_coef=config.training.ent_coef,
        vf_coef=config.training.vf_coef,
        max_grad_norm=config.training.max_grad_norm,
        verbose=config.training.verbose,
        tensorboard_log=os.path.abspath(config.training.log_dir),
    )
    
    return model


def train_bc_policy(
    model: MaskablePPO,
    observations: List[Dict[str, np.ndarray]],
    actions: List[int],
    action_masks: List[Optional[np.ndarray]],
    n_epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
    validation_split: float = 0.2,
    verbose: int = 1,
    tensorboard_log: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Train policy network using behavioral cloning on expert trajectories.
    
    Args:
        model: MaskablePPO model (only policy network will be trained)
        observations: List of observation dictionaries
        actions: List of expert actions
        action_masks: List of action masks (can contain None)
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        validation_split: Fraction of data to use for validation
        verbose: Verbosity level
        tensorboard_log: Optional TensorBoard log directory
        
    Returns:
        Dictionary with training history (train_loss, val_loss, train_acc, val_acc)
    """
    device = model.device
    
    # Create dataset
    dataset = TrajectoryDataset(observations, actions, action_masks)
    
    # Split into train and validation
    dataset_size = len(dataset)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    
    # Create optimizer (only for policy network parameters)
    # Get all parameters from policy network
    policy_params = list(model.policy.parameters())
    optimizer = torch.optim.Adam(policy_params, lr=learning_rate)
    
    # Setup TensorBoard if specified
    from torch.utils.tensorboard import SummaryWriter
    writer = None
    if tensorboard_log:
        os.makedirs(tensorboard_log, exist_ok=True)
        writer = SummaryWriter(tensorboard_log)
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    
    # Early stopping
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None
    
    logger.info(f"Starting BC training:")
    logger.info(f"  Train samples: {train_size}")
    logger.info(f"  Validation samples: {val_size}")
    logger.info(f"  Epochs: {n_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Early stopping patience: {patience}")
    
    # Training loop
    for epoch in range(n_epochs):
        # Training phase
        model.policy.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_batches_processed = 0
        
        for batch in train_loader:
            obs_batch = batch["observation"]
            action_batch = batch["action"].to(device)
            action_mask_batch = batch["action_mask"]
            
            # Move observations to device
            obs_batch_device = {
                k: v.to(device) for k, v in obs_batch.items()
            }
            
            # Move action masks to device if present
            if action_mask_batch is not None:
                action_mask_batch = action_mask_batch.to(device)
            
            # Get policy distribution
            # Extract features
            features = model.policy.extract_features(obs_batch_device)
            
            # Get latent policy
            latent_pi = model.policy.mlp_extractor.forward_actor(features)
            
            # Get action logits
            action_logits = model.policy.action_net(latent_pi)
            
            # Apply action mask if present and filter invalid expert actions
            expert_action_valid = None
            if action_mask_batch is not None:
                # Mask invalid actions by setting logits to -inf
                # action_mask_batch: (batch_size, action_space.n), True for valid actions
                action_logits = action_logits.masked_fill(~action_mask_batch, float('-inf'))
                
                # Check if expert actions are valid
                # Get validity for each expert action in the batch
                batch_indices = torch.arange(action_batch.size(0), device=device)
                expert_action_valid = action_mask_batch[batch_indices, action_batch]
                
                # Filter out invalid expert actions to avoid inf loss
                if not expert_action_valid.all():
                    # Some expert actions are invalid - filter them out
                    valid_mask = expert_action_valid
                    if valid_mask.any():
                        # Keep only valid examples
                        action_logits = action_logits[valid_mask]
                        action_batch = action_batch[valid_mask]
                        action_mask_batch = action_mask_batch[valid_mask]
                    else:
                        # All actions are invalid - skip this batch
                        continue
            
            # Compute loss (cross-entropy)
            # Only compute if we have valid examples
            if action_batch.size(0) > 0:
                loss = F.cross_entropy(action_logits, action_batch)
            else:
                continue
            
            # Compute accuracy
            with torch.no_grad():
                pred_actions = torch.argmax(action_logits, dim=1)
                correct = (pred_actions == action_batch).sum().item()
                train_correct += correct
                train_total += action_batch.size(0)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(policy_params, max_norm=0.5)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches_processed += 1
            
            # Track number of skipped samples due to invalid expert actions
            if expert_action_valid is not None and not expert_action_valid.all():
                skipped = (~expert_action_valid).sum().item()
                if skipped > 0 and verbose > 1:
                    logger.debug(f"Skipped {skipped} samples with invalid expert actions in this batch")
        
        # Validation phase
        model.policy.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batches_processed = 0
        
        with torch.no_grad():
            for batch in val_loader:
                obs_batch = batch["observation"]
                action_batch = batch["action"].to(device)
                action_mask_batch = batch["action_mask"]
                
                # Move observations to device
                obs_batch_device = {
                    k: v.to(device) for k, v in obs_batch.items()
                }
                
                # Move action masks to device if present
                if action_mask_batch is not None:
                    action_mask_batch = action_mask_batch.to(device)
                
                # Get policy distribution
                features = model.policy.extract_features(obs_batch_device)
                latent_pi = model.policy.mlp_extractor.forward_actor(features)
                action_logits = model.policy.action_net(latent_pi)
                
                # Apply action mask if present
                if action_mask_batch is not None:
                    action_logits = action_logits.masked_fill(~action_mask_batch, float('-inf'))
                    
                    # Check if expert actions are valid
                    batch_indices = torch.arange(action_batch.size(0), device=device)
                    expert_action_valid = action_mask_batch[batch_indices, action_batch]
                    
                    # Filter out invalid expert actions
                    if not expert_action_valid.all():
                        valid_mask = expert_action_valid
                        if valid_mask.any():
                            action_logits = action_logits[valid_mask]
                            action_batch = action_batch[valid_mask]
                            action_mask_batch = action_mask_batch[valid_mask]
                        else:
                            continue
                
                # Compute loss (only if we have valid examples)
                if action_batch.size(0) > 0:
                    loss = F.cross_entropy(action_logits, action_batch)
                    val_loss += loss.item()
                    val_batches_processed += 1
                    
                    # Compute accuracy
                    pred_actions = torch.argmax(action_logits, dim=1)
                    correct = (pred_actions == action_batch).sum().item()
                    val_correct += correct
                    val_total += action_batch.size(0)
        
        # Calculate averages (use processed batches count to avoid division by zero)
        avg_train_loss = train_loss / train_batches_processed if train_batches_processed > 0 else float('inf')
        avg_val_loss = val_loss / val_batches_processed if val_batches_processed > 0 else float('inf')
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        # Store history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        # Log progress
        if verbose > 0:
            logger.info(
                f"Epoch {epoch + 1}/{n_epochs}: "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
        
        # TensorBoard logging
        if writer is not None:
            writer.add_scalar("BC/train_loss", avg_train_loss, epoch)
            writer.add_scalar("BC/val_loss", avg_val_loss, epoch)
            writer.add_scalar("BC/train_acc", train_acc, epoch)
            writer.add_scalar("BC/val_acc", val_acc, epoch)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = {
                'policy_state_dict': model.policy.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
            }
            if verbose > 0:
                logger.info(f"  New best validation loss: {best_val_loss:.4f} (epoch {epoch + 1})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose > 0:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs (patience: {patience})")
                break
    
    # Restore best model if early stopping was used
    if best_model_state is not None:
        model.policy.load_state_dict(best_model_state['policy_state_dict'])
        if verbose > 0:
            logger.info(f"Restored best model from epoch {best_model_state['epoch'] + 1} "
                       f"(val_loss: {best_model_state['val_loss']:.4f}, val_acc: {best_model_state['val_acc']:.4f})")
    
    if writer is not None:
        writer.close()
    
    logger.info("BC training completed!")
    
    return history


def train_behavioral_cloning(
    trajectories_dir: str,
    output_model_path: str,
    config: Optional[Config] = None,
    max_trajectories: Optional[int] = None,
    num_envs: int = 1,
    total_timesteps: int = 100000,
    device: Optional[str] = None,
    bc_epochs: int = 10,
    bc_batch_size: int = 256,
    bc_learning_rate: float = 3e-4,
    validation_split: float = 0.2,
) -> MaskablePPO:
    """
    Train agent using behavioral cloning on collected trajectories.
    
    Args:
        trajectories_dir: Directory containing trajectory pickle files
        output_model_path: Path to save trained model
        config: Training configuration (uses DEFAULT_CONFIG if None)
        max_trajectories: Maximum number of trajectories to use
        num_envs: Number of parallel environments for training
        total_timesteps: Total training timesteps
        device: Device to use (None = auto-detect)
        
    Returns:
        Trained MaskablePPO model
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    logger.info("=" * 70)
    logger.info("Training from collected winner trajectories")
    logger.info("=" * 70)
    
    # Load trajectories
    trajectories = load_trajectories(trajectories_dir)
    
    if not trajectories:
        raise ValueError(f"No trajectories found in {trajectories_dir}")
    
    # Prepare training data
    observations, actions, rewards, action_masks = prepare_training_data(
        trajectories, max_trajectories
    )
    
    if not observations or not actions:
        raise ValueError("No valid training data extracted from trajectories")
    
    # Create environment (needed for model initialization)
    # Use a dummy environment for BC training
    # Note: In practice, you might want to use the actual environment for evaluation
    vec_env_raw = PerudoMultiAgentVecEnv(
        num_envs=num_envs,
        num_players=config.game.num_players,
        dice_per_player=config.game.dice_per_player,
        total_dice_values=config.game.total_dice_values,
        max_quantity=config.game.max_quantity,
        history_length=config.game.history_length,
        max_history_length=getattr(config.training, 'transformer_history_length', config.game.history_length),
        opponent_pool=None,  # Not needed for BC
        random_num_players=config.game.random_num_players,
        min_players=config.game.min_players,
        max_players=config.game.max_players,
        reward_config=config.reward,
        training_mode="botplay",  # Use botplay mode
    )
    
    # Wrap with ActionMaskVecMonitor for action masking support (required by MaskablePPO)
    monitor_dir = os.path.join(config.training.log_dir, "monitor")
    os.makedirs(monitor_dir, exist_ok=True)
    vec_env = ActionMaskVecMonitor(vec_env_raw, monitor_dir)
    
    # Create model
    model = create_model_from_config(config, vec_env, device)
    
    logger.info("Model created, starting behavioral cloning training...")
    logger.info(f"  Training steps: {len(observations)}")
    logger.info(f"  BC epochs: {bc_epochs}")
    logger.info(f"  BC batch size: {bc_batch_size}")
    logger.info(f"  BC learning rate: {bc_learning_rate}")
    
    # Train policy network using behavioral cloning
    history = train_bc_policy(
        model=model,
        observations=observations,
        actions=actions,
        action_masks=action_masks,
        n_epochs=bc_epochs,
        batch_size=bc_batch_size,
        learning_rate=bc_learning_rate,
        validation_split=validation_split,
        verbose=config.training.verbose,
        tensorboard_log=os.path.join(os.path.abspath(config.training.log_dir), "bc_training"),
    )
    
    # Log final metrics
    if history["train_loss"]:
        final_train_loss = history["train_loss"][-1]
        final_val_loss = history["val_loss"][-1]
        final_train_acc = history["train_acc"][-1]
        final_val_acc = history["val_acc"][-1]
        
        logger.info("Final BC training metrics:")
        logger.info(f"  Train Loss: {final_train_loss:.4f}, Train Accuracy: {final_train_acc:.4f}")
        logger.info(f"  Val Loss: {final_val_loss:.4f}, Val Accuracy: {final_val_acc:.4f}")
    
    # Save model
    model.save(output_model_path)
    logger.info(f"Model saved to {output_model_path}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train agent from collected trajectories")
    parser.add_argument(
        "--trajectories-dir",
        type=str,
        required=True,
        help="Directory containing trajectory pickle files"
    )
    parser.add_argument(
        "--output-model",
        type=str,
        required=True,
        help="Path to save trained model"
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Maximum number of trajectories to use"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, None = auto-detect)"
    )
    parser.add_argument(
        "--bc-epochs",
        type=int,
        default=10,
        help="Number of BC training epochs"
    )
    parser.add_argument(
        "--bc-batch-size",
        type=int,
        default=256,
        help="Batch size for BC training"
    )
    parser.add_argument(
        "--bc-learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for BC training"
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Train model
    model = train_behavioral_cloning(
        trajectories_dir=args.trajectories_dir,
        output_model_path=args.output_model,
        max_trajectories=args.max_trajectories,
        total_timesteps=args.total_timesteps,
        device=args.device,
        bc_epochs=args.bc_epochs,
        bc_batch_size=args.bc_batch_size,
        bc_learning_rate=args.bc_learning_rate,
        validation_split=args.validation_split,
    )
    
    logger.info("Training completed!")

