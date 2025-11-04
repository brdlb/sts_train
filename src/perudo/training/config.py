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
    policy: str = "MlpPolicy"
    policy_kwargs: Optional[Dict] = field(default_factory=lambda: dict(net_arch=[256, 128, 64]))
    device: Optional[str] = None  # If None, will auto-detect (GPU with CPU fallback)
    learning_rate: float = 1e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2 
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

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
