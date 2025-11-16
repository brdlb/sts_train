"""
Episode tracker for statistics.

This module tracks episode statistics for monitoring and logging.
"""

from typing import Dict, Optional


class EpisodeTracker:
    """
    Tracker for episode statistics.
    
    Responsibilities:
    - Accumulate action statistics
    - Accumulate rewards
    - Generate final episode information
    
    Contract:
    - Supports tracking multiple agents simultaneously
    - Reset clears all statistics
    - Does not affect game logic
    """
    
    def __init__(self):
        """Initialize episode tracker."""
        self.reset()
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.bid_count = 0
        self.challenge_count = 0
        self.believe_count = 0
        self.invalid_action_count = 0
        self.accumulated_reward = 0.0
        self.episode_length = 0
        
        # Per-agent statistics (for multi-agent scenarios)
        self.agent_rewards: Dict[int, float] = {}
        self.agent_lengths: Dict[int, int] = {}
    
    def record_action(self, action_type: str, valid: bool = True) -> None:
        """
        Record an action.
        
        Args:
            action_type: Type of action ('bid', 'challenge', 'believe')
            valid: Whether action was valid
        """
        if not valid:
            self.invalid_action_count += 1
            return
        
        if action_type == "bid":
            self.bid_count += 1
        elif action_type == "challenge":
            self.challenge_count += 1
        elif action_type == "believe":
            self.believe_count += 1
        
        self.episode_length += 1
    
    def accumulate_reward(self, reward: float, agent_id: int = 0) -> None:
        """
        Accumulate reward for agent.
        
        Args:
            reward: Reward to add
            agent_id: ID of agent (default 0 for single agent)
        """
        self.accumulated_reward += reward
        
        if agent_id not in self.agent_rewards:
            self.agent_rewards[agent_id] = 0.0
        self.agent_rewards[agent_id] += reward
    
    def get_episode_info(
        self, 
        winner: Optional[int] = None, 
        final_reward: Optional[float] = None,
        agent_id: int = 0
    ) -> Dict:
        """
        Get final episode information.
        
        Args:
            winner: Winner ID (if game ended)
            final_reward: Final reward (if calculated)
            agent_id: ID of agent to get info for
            
        Returns:
            Dictionary with episode information
        """
        # Use agent-specific reward if available, otherwise use accumulated
        agent_reward = self.agent_rewards.get(agent_id, self.accumulated_reward)
        agent_length = self.agent_lengths.get(agent_id, self.episode_length)
        
        # Use final_reward if provided, otherwise use agent_reward
        reward_to_use = final_reward if final_reward is not None else agent_reward
        
        return {
            "r": float(reward_to_use),
            "l": int(agent_length),
            "bid_count": self.bid_count,
            "challenge_count": self.challenge_count,
            "believe_count": self.believe_count,
            "invalid_action_count": self.invalid_action_count,
            "winner": winner if winner is not None else -1,
        }

