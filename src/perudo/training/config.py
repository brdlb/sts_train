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
class RewardConfig:
    """
    Reward and penalty configuration for training.
    
    Conservative reward scaling (approximately 1/20 of original values) to keep
    episode rewards in a stable range (-20 to +20) for better PPO training stability.
    This prevents reward variance from -500 to +500 that can destabilize learning.
    """

    # Game outcome rewards
    win_reward: float = 5.0  # Reward for winning the game (was 50.0)

    # Dice loss penalties
    dice_lost_penalty: float = -0.5  # Penalty per die lost (was -2.0)

    # Challenge rewards and penalties
    challenge_success_reward: float = 0.1  # Reward for successful challenge (caught bluff) (was 2.0)
    challenge_failure_penalty: float = -0.1  # Penalty for unsuccessful challenge that led to dice loss (was -2.0)

    # Believe rewards and penalties
    believe_success_reward: float = 0.1  # Reward for successful believe call (caught bluff) (was 2.0)
    believe_failure_penalty: float = -0.1  # Penalty for unsuccessful believe call that led to dice loss (was -2.0)

    # Bid-related rewards and penalties
    bid_small_penalty: float = 0  # Small negative reward for bidding to encourage finishing the round
    unsuccessful_bid_penalty: float = -0.1  # Penalty for unsuccessful bid that led to dice loss (was -1.5)

    # Round-end rewards (WARNING: these accumulate over many rounds!)
    round_no_dice_lost_reward: float = 0.05  # Reward for each round in which the agent did not lose a die (was 1.0)
    successful_bluff_reward: float = 0.25  # Reward for successful bluff (bid was correct or never challenged) (was 5.0)

    # Bid defense rewards (when opponent challenges/believes agent's bid and fails)
    defend_bid_reward_challenge: float = 0.1  # Reward for successfully defending bid against challenge (was 2.5)
    defend_bid_reward_believe: float = 0.25  # Reward for successfully defending bid against believe (was 5.0)

    # Dice advantage rewards (WARNING: these are given every step and accumulate!)
    dice_advantage_reward: float = 0.03  # Reward per die advantage over average opponents (was 0.5)
    dice_advantage_max: float = 5.0  # Maximum dice advantage to consider (cap)
    leader_bonus: float = 0.1  # Bonus reward for being the leader (having more dice than all opponents) (was 2.0)

    # Invalid action penalty
    invalid_action_penalty: float = -0.05  # Penalty for invalid action (action not allowed by rules) (was -1.0)


@dataclass
class TrainingConfig:
    """Training configuration."""

    # PPO parameters (optimized for new transformer architecture: 3 layers, 20% fewer params)
    policy: str = "MultiInputPolicy"  # Use MultiInputPolicy for Dict observation space
    policy_kwargs: Optional[Dict] = None  # Will be set based on transformer config
    device: Optional[str] = None  # If None, will auto-detect (GPU with CPU fallback)
    opponent_device: Optional[str] = "cpu"  
    learning_rate: float = 3.0e-4  
    n_steps: int = 2048  
    batch_size: int = 256  
    n_epochs: int = 10  
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01 
    vf_coef: float = 0.75  
    max_grad_norm: float = 0.3  
    
    # Transformer parameters (optimized for sequence length 20)
    transformer_features_dim: int = 192  # Reduced from 256 for better efficiency
    transformer_num_layers: int = 3  # Increased from 2 for better expressiveness
    transformer_num_heads: int = 4  # Optimal for embed_dim=96 (96/4=24 per head)
    transformer_embed_dim: int = 96  # Reduced from 128 (sufficient for seq_len=20)
    transformer_dim_feedforward: int = 384  # Reduced proportionally (96 * 4 = 384)
    transformer_history_length: int = 20
    transformer_dropout: float = 0.1  # Explicit dropout parameter

    # Training parameters
    num_envs: int = 16  # Number of parallel environments (tables)
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
