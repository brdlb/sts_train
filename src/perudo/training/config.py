"""
Configuration for training Perudo agents.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any


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
class RewardConfig:
    """
    Reward and penalty configuration for training.
    
    Using sparse reward approach: primarily reward for winning, minimal intermediate rewards.
    This prevents reward hacking and allows agent to focus on the primary objective.
    Intermediate rewards are set to 0.0 but kept in structure for easy re-enabling if needed.
    """

    win_reward: float = 2
    win_dice_bonus: float = 0
    lose_penalty: float = -0.5

    dice_lost_penalty: float = 0

    challenge_success_reward: float = 0.1
    challenge_failure_penalty: float = 0

    believe_success_reward: float = 0.1
    believe_failure_penalty: float = 0

    bid_small_penalty: float = 0
    unsuccessful_bid_penalty: float = -0.1

    bid_minimal_bonus: float = 0
    bid_excess_penalty: float = 0
    bid_excess_threshold: int = 1

    round_no_dice_lost_reward: float = 0.0
    successful_bluff_reward: float = 0.0

    defend_bid_reward_challenge: float = 0.0
    defend_bid_reward_believe: float = 0.0

    dice_advantage_reward: float = 0.0
    dice_advantage_max: float = 0.0
    leader_bonus: float = 0.0

    invalid_action_penalty: float = -0.00


@dataclass
class TrainingConfig:
    """Training configuration."""

    policy: str = "MultiInputPolicy"
    policy_kwargs: Optional[Dict] = None
    device: Optional[str] = None
    opponent_device: Optional[str] = "cuda"
    learning_rate: float = 1.3e-4
    n_steps: int = 20480
    batch_size: int = 2560
    n_epochs: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.98
    clip_range: float = 0.08
    ent_coef: float = 0.08
    vf_coef: float = 0.25
    max_grad_norm: float = 0.3
    
    adaptive_entropy: bool = True
    entropy_threshold_low: float = -0.45
    entropy_threshold_high: float = -0.35
    entropy_adjustment_rate: float = 0.02
    entropy_max_coef: float = 0.15
    
    transformer_features_dim: int = 256
    transformer_num_layers: int = 3
    transformer_num_heads: int = 8
    transformer_embed_dim: int = 256
    transformer_dim_feedforward: int = 1024
    transformer_history_length: int = 24
    transformer_dropout: float = 0.15

    num_envs: int = 4
    total_timesteps: int = 6_000_000
    save_freq: int = 100_000
    eval_freq: int = 50_000
    eval_episodes: int = 10

    log_dir: str = "logs"
    model_dir: str = "models"
    tb_log_name: Optional[str] = None

    use_rule_based_opponents: bool = True
    training_mode: str = "mixed"
    bot_difficulty_distribution: Dict[str, float] = field(
        default_factory=lambda: {"EASY": 0.33, "MEDIUM": 0.34, "HARD": 0.33}
    )  # Distribution of bot difficulty levels
    mixed_mode_ratio: float = 0.7  # Ratio of botplay in mixed mode (0.0-1.0)
    allowed_bot_personalities: Optional[List[str]] = None #field(
    
    
     #   default_factory=lambda: ["CONSERVATIVE"] )  # List of allowed bot personality keys (e.g., ["CONSERVATIVE"]). If None, all bots are allowed.

    # Trajectory collection
    collect_trajectories: bool = False  # Enable collection and saving of winner trajectories (for imitation learning)
    
    # Other
    verbose: int = 1
    seed: Optional[int] = None
    debug_moves: bool = True  # Enable detailed move logging (forces num_envs=1)


@dataclass
class Config:
    """General configuration."""

    game: GameConfig = None
    training: TrainingConfig = None
    reward: RewardConfig = None

    def __post_init__(self):
        """Initialize default values."""
        if self.game is None:
            self.game = GameConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.reward is None:
            self.reward = RewardConfig()


# Default configuration
DEFAULT_CONFIG = Config()
