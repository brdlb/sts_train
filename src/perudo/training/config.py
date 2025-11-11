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

    # Game outcome rewards
    win_reward: float = 2  # Reward for winning the game (sparse reward - main objective)
    win_dice_bonus: float = 0.1 # Bonus per remaining die when winning the game
    lose_penalty: float = 0  # Penalty for losing the game (negative reward to distinguish from win)

    # Dice loss penalties
    dice_lost_penalty: float = -0.5 # Minimal penalty per die lost (for training stability)

    # Challenge rewards and penalties (DISABLED - set to 0.0 for sparse rewards)
    challenge_success_reward: float = 0.1  # Reward for successful challenge (caught bluff) - DISABLED
    challenge_failure_penalty: float = -0.05  # Penalty for unsuccessful challenge that led to dice loss - DISABLED

    # Believe rewards and penalties (DISABLED - set to 0.0 for sparse rewards)
    believe_success_reward: float = 0.1  # Reward for successful believe call (caught bluff) - DISABLED
    believe_failure_penalty: float = -0.05  # Penalty for unsuccessful believe call that led to dice loss - DISABLED

    # Bid-related rewards and penalties (DISABLED - set to 0.0 for sparse rewards)
    bid_small_penalty: float = -0.01  # Small negative reward for bidding to encourage finishing the round - DISABLED
    unsuccessful_bid_penalty: float = -0.1  # Penalty for unsuccessful bid that led to dice loss - DISABLED


    round_no_dice_lost_reward: float = 0.0  # Reward for each round in which the agent did not lose a die - DISABLED
    successful_bluff_reward: float = 0.0  # Reward for successful bluff (bid was correct or never challenged) - DISABLED

    # Bid defense rewards (DISABLED - set to 0.0 for sparse rewards)
    # When opponent challenges/believes agent's bid and fails
    defend_bid_reward_challenge: float = 0.01  # Reward for successfully defending bid against challenge - DISABLED
    defend_bid_reward_believe: float = 0.01  # Reward for successfully defending bid against believe - DISABLED

    # Dice advantage rewards (DISABLED - set to 0.0 for sparse rewards)
    # WARNING: these are given every step and accumulate, can cause reward hacking
    dice_advantage_reward: float = 0.0  # Reward per die advantage over average opponents - DISABLED
    dice_advantage_max: float = 0.0  # Maximum dice advantage to consider (cap) - kept for structure
    leader_bonus: float = 0.0  # Bonus reward for being the leader (having more dice than all opponents) - DISABLED

    # Invalid action penalty (kept minimal to guide learning, but not too harsh)
    invalid_action_penalty: float = -0.01  # Minimal penalty for invalid action (helps avoid invalid actions)


@dataclass
class TrainingConfig:
    """Training configuration."""

    # PPO parameters (optimized to address critical clip_fraction issue)
    policy: str = "MultiInputPolicy"  # Use MultiInputPolicy for Dict observation space
    policy_kwargs: Optional[Dict] = None  # Will be set based on transformer config
    device: Optional[str] = None  # If None, will auto-detect (GPU with CPU fallback)
    opponent_device: Optional[str] = "cpu"  
    learning_rate: float = 4.0e-5  # Reduced for stability with enhanced critic architecture
    n_steps: int = 8192
    batch_size: int = 512
    n_epochs: int = 10  # Reduced to prevent overfitting on batch
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.01  # Critically reduced to address high clip_fraction
    ent_coef: float = 0.15  # Slightly increased for better exploration
    vf_coef: float = 0.5  # Standard value; with enhanced critic architecture (vf=[256, 128]), high coefficient can cause overfitting
    max_grad_norm: float = 0.5  # Increased for less aggressive gradient clipping
    
    # Adaptive entropy coefficient parameters
    adaptive_entropy: bool = True  # Enable adaptive entropy coefficient adjustment
    entropy_threshold_low: float = -3.48  # Slightly lower for earlier response
    entropy_threshold_high: float = -3.32  # Slightly higher for wider range
    entropy_adjustment_rate: float = 0.008  # Rate of ent_coef adjustment per update (slower)
    entropy_max_coef: float = 0.25  # Maximum allowed ent_coef value to prevent excessive exploration  
    
    # Transformer parameters (optimized for sequence length 40)
    transformer_features_dim: int = 256  # Increased to handle richer feature information
    transformer_num_layers: int = 3  # Increased from 2 for better expressiveness
    transformer_num_heads: int = 8  # Increased for better attention analysis (128/8=16 per head)
    transformer_embed_dim: int = 128  # Increased to provide more "space" for representing each move
    transformer_dim_feedforward: int = 512  # Standard practice: 128 * 4 = 512
    transformer_history_length: int = 40  # Increased to cover full game history
    transformer_dropout: float = 0.1  # Explicit dropout parameter

    # Training parameters
    num_envs: int = 1  # Number of parallel environments (tables)
    total_timesteps: int = 1_000_000
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
