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
    win_dice_bonus: float = 0 # Bonus per remaining die when winning the game
    lose_penalty: float = -0.5  # Penalty for losing the game (negative reward to distinguish from win)

    # Dice loss penalties
    dice_lost_penalty: float = 0 # Minimal penalty per die lost (for training stability)

    # Challenge rewards and penalties (DISABLED - set to 0.0 for sparse rewards)
    challenge_success_reward: float = 0.1  # Reward for successful challenge (caught bluff) - DISABLED
    challenge_failure_penalty: float = 0  # Penalty for unsuccessful challenge that led to dice loss - DISABLED

    # Believe rewards and penalties (DISABLED - set to 0.0 for sparse rewards)
    believe_success_reward: float = 0.1  # Reward for successful believe call (caught bluff) - DISABLED
    believe_failure_penalty: float = 0  # Penalty for unsuccessful believe call that led to dice loss - DISABLED

    # Bid-related rewards and penalties (DISABLED - set to 0.0 for sparse rewards)
    bid_small_penalty: float = 0 # Small negative reward for bidding to encourage finishing the round - DISABLED
    unsuccessful_bid_penalty: float = -0.1 # Penalty for unsuccessful bid that led to dice loss - DISABLED
    
    # Minimal bid incentives (ENABLED - gently encourages minimal bids)
    bid_minimal_bonus: float = 0  # Small bonus for making minimal valid bid
    bid_excess_penalty: float = 0 # Small penalty for exceeding minimal bid (proportional to excess)
    bid_excess_threshold: int = 1  # Penalty applies only if excess >= threshold


    round_no_dice_lost_reward: float = 0.0  # Reward for each round in which the agent did not lose a die - DISABLED
    successful_bluff_reward: float = 0.0  # Reward for successful bluff (bid was correct or never challenged) - DISABLED

    # Bid defense rewards (DISABLED - set to 0.0 for sparse rewards)
    # When opponent challenges/believes agent's bid and fails
    defend_bid_reward_challenge: float = 0.0  # Reward for successfully defending bid against challenge - DISABLED
    defend_bid_reward_believe: float = 0.0  # Reward for successfully defending bid against believe - DISABLED

    # Dice advantage rewards (DISABLED - set to 0.0 for sparse rewards)
    # WARNING: these are given every step and accumulate, can cause reward hacking
    dice_advantage_reward: float = 0.0  # Reward per die advantage over average opponents - DISABLED
    dice_advantage_max: float = 0.0  # Maximum dice advantage to consider (cap) - kept for structure
    leader_bonus: float = 0.0  # Bonus reward for being the leader (having more dice than all opponents) - DISABLED

    # Invalid action penalty (kept minimal to guide learning, but not too harsh)
    invalid_action_penalty: float = -0.00  # Minimal penalty for invalid action (helps avoid invalid actions)


@dataclass
class TrainingConfig:
    """Training configuration."""

    # PPO parameters - Optimized for compact model based on training analysis
    # Analysis: clip_fraction 0.15-0.25 indicates good clipping, but need stability improvements
    policy: str = "MultiInputPolicy"  # Use MultiInputPolicy for Dict observation space
    policy_kwargs: Optional[Dict] = None  # Will be set based on transformer config
    device: Optional[str] = None 
    opponent_device: Optional[str] = "cuda"  
    learning_rate: float = 2.0e-4  # Conservative learning rate for compact model
    n_steps: int = 6144  # More frequent updates (effective buffer size: 6144 × num_envs)
    batch_size: int = 1024  # Better for compact model (6144×4/1024=24 batches)
    n_epochs: int = 6  # More frequent updates while maintaining training quality
    gamma: float = 0.99  # Standard discount factor
    gae_lambda: float = 0.95  # Standard GAE parameter
    clip_range: float = 0.2  # Standard PPO value for stability
    ent_coef: float = 0.05 
    vf_coef: float = 0.75  # Balanced value
    max_grad_norm: float = 0.75  # Improves gradient stability 
    
    # Adaptive entropy coefficient parameters
    # Analysis: entropy_loss shows policy becoming too deterministic around 40k steps
    # Adjusted thresholds to maintain better exploration-exploitation balance
    adaptive_entropy: bool = True  # Enable adaptive entropy coefficient adjustment
    entropy_threshold_low: float = -3.5  # Lower threshold - increase ent_coef when entropy too low
    entropy_threshold_high: float = -3.0  # Higher threshold - decrease ent_coef when entropy too high
    entropy_adjustment_rate: float = 0.01  # Faster adjustment to respond quickly to entropy issues
    entropy_max_coef: float = 0.1  # Aligned with new ent_coef=0.01  
    
    # Transformer parameters (optimized for sequence length 12)
    transformer_features_dim: int = 128  # Output feature dimension
    transformer_num_layers: int = 1  # Single layer sufficient for very short sequences (12 events)
    transformer_num_heads: int = 4  # Reduced to 4 for efficient attention on sequences (12/4=3 events per head)
    transformer_embed_dim: int = 96  # Sufficient for short sequence representations
    transformer_dim_feedforward: int = 384  # Standard practice: 96 * 4 = 384
    transformer_history_length: int = 12  # Reduced to 12 events for maximum efficiency and speed
    transformer_dropout: float = 0.1  # Explicit dropout parameter

    # Training parameters
    num_envs: int = 4  # Number of parallel environments (tables)
    total_timesteps: int = 5_000_000
    save_freq: int = 100_000  # Save model every N steps
    eval_freq: int = 50_000  # Evaluate model every N steps
    eval_episodes: int = 10  # Number of episodes for evaluation

    # Paths
    log_dir: str = "logs"
    model_dir: str = "models"
    tb_log_name: Optional[str] = None

    # Rule-based opponent configuration
    use_rule_based_opponents: bool = True  # Use rule-based bots as opponents
    training_mode: str = "botplay"  # 'selfplay', 'botplay', or 'mixed'
    bot_difficulty_distribution: Dict[str, float] = field(
        default_factory=lambda: {"EASY": 0.33, "MEDIUM": 0.34, "HARD": 0.33}
    )  # Distribution of bot difficulty levels
    mixed_mode_ratio: float = 0.5  # Ratio of botplay in mixed mode (0.0-1.0)
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
