"""
Reward management module for Perudo vectorized environment.

Handles reward accumulation, final reward calculation, and step reward computation
for VecMonitor compatibility.

"""

from typing import List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..training.config import RewardConfig
else:
    RewardConfig = Any


class RewardManager:
    """
    Manages reward accumulation and calculation for learning agents.
    
    """
    
    def __init__(self, num_envs: int, reward_config: Optional[RewardConfig] = None):
        """
        Initialize reward manager.
        
        Args:
            num_envs: Number of parallel environments
            reward_config: Reward configuration (uses DEFAULT_CONFIG if not provided)
        """
        self.num_envs = num_envs
        self.reward_config = reward_config
        

        self.vecmonitor_accumulated_reward: List[float] = [0.0] * num_envs
        
        self.deferred_reward: List[float] = [0.0] * num_envs
        
        # Validate reward configuration if provided
        if reward_config is not None:
            self._validate_reward_config(reward_config)
    
    def reset_episode(self, env_idx: int) -> None:
        """
        Reset reward tracking for an episode.
        
        Args:
            env_idx: Environment index
        """
        self.vecmonitor_accumulated_reward[env_idx] = 0.0
        self.deferred_reward[env_idx] = 0.0
    
    def accumulate_reward(self, env_idx: int, reward: float) -> None:
        """
        Accumulate reward for learning agent during episode.
        
        Args:
            env_idx: Environment index
            reward: Reward to accumulate
        """
        self.vecmonitor_accumulated_reward[env_idx] += reward
    
    def get_accumulated_reward(self, env_idx: int) -> float:
        """
        Get accumulated reward for an environment.
        
        Args:
            env_idx: Environment index
            
        Returns:
            Accumulated reward (does not include dice lost penalties or deferred rewards)
        """
        return self.vecmonitor_accumulated_reward[env_idx]
    
    def accumulate_deferred_reward(self, env_idx: int, reward: float) -> None:
        """
        Accumulate deferred reward for learning agent.
        
        Deferred rewards are rewards that are earned during opponent turns
        (e.g., when agent successfully defends their bid after opponent challenges).
        These are accumulated separately and included in final reward calculation.
        
        Args:
            env_idx: Environment index
            reward: Deferred reward to accumulate
        """
        self.deferred_reward[env_idx] += reward
    
    def get_deferred_reward(self, env_idx: int) -> float:
        """
        Get accumulated deferred reward for an environment.
        
        Args:
            env_idx: Environment index
            
        Returns:
            Accumulated deferred reward
        """
        return self.deferred_reward[env_idx]
    
    def calculate_final_reward(
        self, 
        env_idx: int,
        env: Any, 
        learning_agent_id: int, 
        reward_config: Optional[RewardConfig] = None
    ) -> float:
        """
        Calculate final reward for learning agent at episode end.
        
        Args:
            env_idx: Environment index (for accessing accumulated rewards)
            env: Environment instance
            learning_agent_id: ID of learning agent (usually 0)
            reward_config: Reward configuration (uses self.reward_config if not provided)
            
        Returns:
            Final reward including intermediate rewards, dice lost penalty, and win/lose bonuses
            

        """
        if reward_config is None:
            reward_config = self.reward_config
        
        if reward_config is None:
            return self.get_accumulated_reward(env_idx)  # Fallback to accumulated if no config
        

        episode_accumulated_reward = self.get_accumulated_reward(env_idx)
        
        # Add deferred rewards (e.g., when agent successfully defends bid)
        deferred_reward = self.get_deferred_reward(env_idx)
        episode_accumulated_reward += deferred_reward
        
        initial_dice = env.dice_per_player
        # Validate final dice count before accessing
        if (learning_agent_id < 0 or 
            learning_agent_id >= len(env.game_state.player_dice_count) or
            env.game_state.player_dice_count[learning_agent_id] is None):
            # Invalid final dice count, use 0 as fallback (no penalty applied)
            final_dice = 0
        else:
            final_dice = env.game_state.player_dice_count[learning_agent_id]
            # Ensure final_dice is non-negative
            final_dice = max(0, final_dice)
        
        # Ensure initial_dice is valid (should always be positive, but check for safety)
        initial_dice = max(0, initial_dice)
        
        # Calculate total dice lost (cannot be negative)
        total_dice_lost = max(0, initial_dice - final_dice)
        total_dice_lost_penalty = reward_config.dice_lost_penalty * total_dice_lost
        
        # Add dice lost penalty to accumulated reward
        episode_accumulated_reward += total_dice_lost_penalty
        
        # Add win/lose bonuses
        # CRITICAL: Final episode rewards are applied here ONLY, not in calculate_reward().
        # This ensures win_reward and win_dice_bonus are counted exactly once, regardless of
        # whether the game ended on learning agent's turn or opponent's turn.
        winner = env.game_state.winner if hasattr(env.game_state, "winner") and env.game_state.winner is not None else -1
        
        if winner == learning_agent_id:
            # Learning agent won: add win_reward and win_dice_bonus
            final_reward = episode_accumulated_reward + reward_config.win_reward
            # Add bonus for remaining dice if winner is valid and bonus is configured
            # Validate winner index and dice count before accessing
            if (winner >= 0 and 
                winner < len(env.game_state.player_dice_count) and
                env.game_state.player_dice_count[winner] is not None and
                reward_config.win_dice_bonus > 0):
                winner_dice_count = env.game_state.player_dice_count[winner]
                # Additional safety check: ensure dice count is non-negative
                if winner_dice_count >= 0:
                    final_reward += reward_config.win_dice_bonus * winner_dice_count
        else:
            # Learning agent lost: add lose_penalty
            final_reward = episode_accumulated_reward + reward_config.lose_penalty
        
        return final_reward
    
    def calculate_step_reward(
        self, 
        env_idx: int, 
        done: bool, 
        final_episode_reward: Optional[float] = None,
        current_step_reward: float = 0.0,
        is_learning_agent: bool = True
    ) -> float:
        """
        Calculate step reward for VecMonitor.
        
        Args:
            env_idx: Environment index
            done: Whether episode is done
            final_episode_reward: Final episode reward (required if done=True)
            current_step_reward: Reward from current step (for non-done steps)
            is_learning_agent: Whether current step is from learning agent
            
        Returns:
            Step reward for VecMonitor
            
        """
        if done:
            # Episode ended: calculate step_reward as difference between final reward and accumulated reward
            if final_episode_reward is not None:
                step_reward = final_episode_reward - self.vecmonitor_accumulated_reward[env_idx]
            else:
                print(f"Warning: calculate_step_reward called with done=True but final_episode_reward=None for env_idx={env_idx}. "
                      f"Using accumulated reward {self.vecmonitor_accumulated_reward[env_idx]} as fallback.")
                step_reward = self.vecmonitor_accumulated_reward[env_idx]
                # Don't reset accumulated reward yet - caller may need to recalculate
            # Reset accumulated reward for next episode
            self.vecmonitor_accumulated_reward[env_idx] = 0.0
        else:
            # During episode: return reward for learning agent (agent 0), 0.0 for opponent
            if is_learning_agent:
                # Learning agent's step, return actual reward (already accumulated)
                step_reward = current_step_reward
            else:
                # Opponent's step, learning agent gets 0 reward
                step_reward = 0.0
        
        return step_reward
    
    def _validate_reward_config(self, reward_config: RewardConfig) -> None:
        """
        Validate reward configuration.
        
        Args:
            reward_config: Reward configuration to validate
            
        Raises:
            ValueError: If reward configuration is invalid
        """
        # Validate that win_reward is greater than lose_penalty
        if reward_config.win_reward <= reward_config.lose_penalty:
            raise ValueError(
                f"Invalid reward config: win_reward ({reward_config.win_reward}) "
                f"must be greater than lose_penalty ({reward_config.lose_penalty})"
            )

