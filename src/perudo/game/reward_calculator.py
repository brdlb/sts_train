"""
Reward calculator for reinforcement learning.

This module handles all reward calculations for the Perudo game environment.
"""

from typing import Dict, List, Optional, Any
from .interfaces import IRewardCalculator
from ..training.config import RewardConfig, DEFAULT_CONFIG
from ..utils.helpers import calculate_reward


class RewardCalculator(IRewardCalculator):
    """
    Calculator for reward computation in reinforcement learning.
    
    Responsibilities:
    - Calculate rewards for actions (bid, challenge, believe)
    - Calculate rewards for round end
    - Calculate final rewards
    - Calculate bonuses for dice advantage
    
    Contract:
    - All calculations are deterministic (same inputs -> same result)
    - Configuration is set at initialization
    - Does not store state between calls
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        """
        Initialize reward calculator.
        
        Args:
            config: Reward configuration (uses DEFAULT_CONFIG.reward if not provided)
        """
        self.config = config if config is not None else DEFAULT_CONFIG.reward
    
    def calculate_step_reward(
        self,
        action_type: str,
        action_result: Dict[str, Any],
        game_state,
        learning_agent_id: int
    ) -> float:
        """
        Calculate reward for one step.
        
        Args:
            action_type: Action type ('bid', 'challenge', 'believe')
            action_result: Action result dictionary containing:
                - success: bool (for challenge/believe)
                - dice_lost: int
                - game_over: bool
                - winner: int (if game_over)
                - winner_dice_count: int (optional, if game_over and winner == learning_agent_id)
                - player_id: int (ID of player who took action)
            game_state: Current game state
            learning_agent_id: ID of learning agent
            
        Returns:
            Step reward (may be negative)
        """
        player_id = action_result.get("player_id", learning_agent_id)
        game_over = action_result.get("game_over", False)
        winner = action_result.get("winner", -1)
        dice_lost = action_result.get("dice_lost", 0)
        challenge_success = action_result.get("challenge_success")
        believe_success = action_result.get("believe_success")
        winner_dice_count = action_result.get("winner_dice_count")
        
        # Use existing calculate_reward function
        reward = calculate_reward(
            action_type=action_type,
            game_over=game_over,
            winner=winner,
            player_id=player_id,
            challenge_success=challenge_success,
            believe_success=believe_success,
            dice_lost=dice_lost,
            reward_config=self.config,
            winner_dice_count=winner_dice_count,
        )
        
        # Add small penalty for bidding (to encourage finishing the round)
        if action_type == "bid" and player_id == learning_agent_id:
            reward += self.config.bid_small_penalty
        
        return reward
    
    def calculate_final_reward(
        self, 
        game_state,
        learning_agent_id: int, 
        accumulated_reward: float
    ) -> float:
        """
        Calculate final reward for episode.
        
        Args:
            game_state: Final game state
            learning_agent_id: ID of learning agent
            accumulated_reward: Accumulated reward for episode
            
        Returns:
            Final reward (including win/lose bonuses)
        """
        winner = game_state.winner if hasattr(game_state, "winner") and game_state.winner is not None else -1
        
        if winner == learning_agent_id:
            final_reward = accumulated_reward + self.config.win_reward
            if winner >= 0 and winner < len(game_state.player_dice_count):
                winner_dice_count = game_state.player_dice_count[winner]
                if winner_dice_count is not None and self.config.win_dice_bonus > 0:
                    final_reward += self.config.win_dice_bonus * winner_dice_count
        else:
            final_reward = accumulated_reward + self.config.lose_penalty
        
        return final_reward
    
    def calculate_round_end_reward(
        self,
        agent_lost_dice: bool,
        agent_bid_this_round: bool,
        learning_agent_id: int
    ) -> float:
        """
        Calculate reward for round end.
        
        Args:
            agent_lost_dice: Whether agent lost dice this round
            agent_bid_this_round: Whether agent made a bid this round
            learning_agent_id: ID of learning agent
            
        Returns:
            Round end reward
        """
        reward = 0.0
        
        # Penalty for unsuccessful bid that led to dice loss
        if agent_lost_dice and agent_bid_this_round:
            reward += self.config.unsuccessful_bid_penalty
        
        # Reward for each round in which the agent did not lose a die
        if not agent_lost_dice:
            reward += self.config.round_no_dice_lost_reward
            
            if agent_bid_this_round:
                # Agent successfully bluffed (bid was correct or never challenged)
                reward += self.config.successful_bluff_reward
        
        return reward
    
    def calculate_dice_advantage_reward(
        self,
        agent_dice: int,
        opponent_dice_counts: List[int],
        learning_agent_id: int
    ) -> float:
        """
        Calculate reward for dice advantage.
        
        Args:
            agent_dice: Number of dice agent has
            opponent_dice_counts: List of opponent dice counts
            learning_agent_id: ID of learning agent
            
        Returns:
            Dice advantage reward
        """
        if not opponent_dice_counts:
            return 0.0
        
        avg_opponent_dice = sum(opponent_dice_counts) / len(opponent_dice_counts)
        dice_advantage = agent_dice - avg_opponent_dice
        
        reward = 0.0
        
        # Reward per die advantage (capped at reasonable maximum)
        if dice_advantage > 0:
            max_advantage = self.config.dice_advantage_max if self.config.dice_advantage_max > 0 else float('inf')
            capped_advantage = min(dice_advantage, max_advantage)
            reward += self.config.dice_advantage_reward * capped_advantage
        
        # Bonus reward for being leader (having more dice than all opponents)
        max_opponent_dice = max(opponent_dice_counts)
        if agent_dice > max_opponent_dice:
            # Additional reward for being the leader
            reward += self.config.leader_bonus
        
        return reward

