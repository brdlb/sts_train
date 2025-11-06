"""
Configuration for training Perudo agents.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class GameConfig:
    """Game configuration."""

    num_players: int = 4
    random_num_players: bool = False  # If True, randomly select num_players in each episode
    min_players: int = 3  # Minimum number of players (used when random_num_players=True)
    max_players: int = 8  # Maximum number of players (used when random_num_players=True)
    dice_per_player: int = 5
    total_dice_values: int = 6
    max_quantity: int = 30
    history_length: int = 10


@dataclass
class TrainingConfig:
    """Training configuration."""

    # PPO parameters (optimized for new transformer architecture: 3 layers, 20% fewer params)
    policy: str = "MultiInputPolicy"  # Use MultiInputPolicy for Dict observation space
    policy_kwargs: Optional[Dict] = None  # Will be set based on transformer config
    device: Optional[str] = None  # If None, will auto-detect (GPU with CPU fallback)
    opponent_device: Optional[str] = "cpu"  # Device for opponent models (CPU recommended to avoid GPU overhead)
    learning_rate: float = 2.0e-4  # Increased (fewer params + LayerNorm allow higher LR)
    n_steps: int = 2048  # Steps to collect before update (for 1 env, 4 players: effective batch = 8192)
    batch_size: int = 256  # Adjusted for 1 env, 4 players: effective batch = 8192, resulting in 32 mini-batches
    n_epochs: int = 8  # Increased for better data utilization with smaller effective batch
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_range: float = 0.12  # Reduced (more conservative for 3-layer transformer)
    ent_coef: float = 0.02  # Increased to prevent premature convergence (was 0.01)
    vf_coef: float = 0.75  # Increased to better train value function (was 0.5)
    max_grad_norm: float = 0.3  # Reduced (stricter clipping for deeper 3-layer network)
    
    # Transformer parameters (optimized for sequence length 20)
    transformer_features_dim: int = 192  # Reduced from 256 for better efficiency
    transformer_num_layers: int = 3  # Increased from 2 for better expressiveness
    transformer_num_heads: int = 4  # Optimal for embed_dim=96 (96/4=24 per head)
    transformer_embed_dim: int = 96  # Reduced from 128 (sufficient for seq_len=20)
    transformer_dim_feedforward: int = 384  # Reduced proportionally (96 * 4 = 384)
    transformer_history_length: int = 20
    transformer_dropout: float = 0.1  # Explicit dropout parameter

    # Training parameters
    num_envs: int = 1  # Number of parallel environments (tables)
    total_timesteps: int = 10_000_000
    save_freq: int = 100_000  # Save model every N steps
    eval_freq: int = 50_000  # Evaluate model every N steps
    eval_episodes: int = 10  # Number of episodes for evaluation

    # Paths
    log_dir: str = "logs"
    model_dir: str = "models"
    tb_log_name: Optional[str] = None

    # Other
    verbose: int = 1
    seed: Optional[int] = None
    debug_moves: bool = False  # Enable detailed move logging (forces num_envs=1)


@dataclass
class Config:
    """General configuration."""

    game: GameConfig = None
    training: TrainingConfig = None

    def __post_init__(self):
        """Initialize default values."""
        if self.game is None:
            self.game = GameConfig()
        if self.training is None:
            self.training = TrainingConfig()


# Default configuration
DEFAULT_CONFIG = Config()
