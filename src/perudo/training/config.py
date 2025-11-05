"""
Configuration for training Perudo agents.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class GameConfig:
    """Game configuration."""

    num_players: int = 4
    dice_per_player: int = 5
    total_dice_values: int = 6
    max_quantity: int = 30
    history_length: int = 10


@dataclass
class TrainingConfig:
    """Training configuration."""

    # PPO parameters
    policy: str = "MultiInputPolicy"  # Use MultiInputPolicy for Dict observation space
    policy_kwargs: Optional[Dict] = None  # Will be set based on transformer config
    device: Optional[str] = None  # If None, will auto-detect (GPU with CPU fallback)
    learning_rate: float = 1.5e-4  # Reduced for stability (prevents catastrophic forgetting)
    n_steps: int = 1024  # Increased for more stable updates (was 512, too small)
    batch_size: int = 128
    n_epochs: int = 6  # Increased to better train value function (was 4)
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_range: float = 0.15  # Reduced for more conservative policy updates (prevents large approx_kl)
    ent_coef: float = 0.02  # Increased to prevent premature convergence (was 0.01)
    vf_coef: float = 0.75  # Increased to better train value function (was 0.5)
    max_grad_norm: float = 0.5
    
    # Transformer parameters
    transformer_features_dim: int = 256
    transformer_num_layers: int = 2
    transformer_num_heads: int = 4
    transformer_embed_dim: int = 128
    transformer_dim_feedforward: int = 512
    transformer_history_length: int = 20

    # Training parameters
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
