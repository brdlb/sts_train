"""
Abstract interfaces (contracts) for game components.

This module defines the contracts that all game components must follow,
ensuring clear separation of concerns and testability.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class IGameController(ABC):
    """
    Interface for managing game process.
    
    Contract:
    - Methods do not modify game_state directly (only through public GameState methods)
    - All methods are thread-safe (do not store mutable state)
    - Return either a new value or None if operation failed
    """
    
    @abstractmethod
    def skip_to_next_active_player(self, game_state, current_player: int) -> int:
        """
        Find next active player (with dice).
        
        Args:
            game_state: Current game state
            current_player: ID of current player
            
        Returns:
            ID of next active player
            
        Raises:
            ValueError: If no active players remain
        """
        pass
    
    @abstractmethod
    def handle_round_end(
        self, 
        game_state, 
        loser_id: int, 
        winner_id: Optional[int] = None
    ) -> None:
        """
        Handle round end.
        
        Args:
            game_state: Current game state (will be modified)
            loser_id: ID of player who lost the round
            winner_id: ID of player starting next round (if None, uses loser_id)
        """
        pass
    
    @abstractmethod
    def force_initial_bid_if_needed(
        self, 
        game_state, 
        player_id: int, 
        action: int, 
        max_quantity: int
    ) -> Tuple[int, bool]:
        """
        Check and force replace action with bid if this is round start.
        
        Args:
            game_state: Current game state
            player_id: ID of player
            action: Current action
            max_quantity: Maximum quantity for bids
            
        Returns:
            Tuple (new_action, was_replaced)
        """
        pass


class IRewardCalculator(ABC):
    """
    Interface for reward calculation.
    
    Contract:
    - All calculations are deterministic (same inputs -> same result)
    - Configuration is set at initialization
    - Does not store state between calls
    """
    
    @abstractmethod
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
            action_result: Action result (success, dice_lost, game_over, etc.)
            game_state: Current game state
            learning_agent_id: ID of learning agent
            
        Returns:
            Step reward (may be negative)
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass


class IObservationBuilder(ABC):
    """
    Interface for building observations.
    
    Contract:
    - Observations always match observation_space
    - Action masks always have correct size
    - Thread-safe
    """
    
    @abstractmethod
    def build_observation(
        self, 
        game_state, 
        player_id: int
    ) -> Dict[str, np.ndarray]:
        """
        Build observation for player.
        
        Args:
            game_state: Current game state
            player_id: ID of player
            
        Returns:
            Observation in Dict[str, np.ndarray] format
        """
        pass
    
    @abstractmethod
    def get_action_mask(
        self, 
        game_state, 
        player_id: int
    ) -> np.ndarray:
        """
        Get mask of available actions.
        
        Args:
            game_state: Current game state
            player_id: ID of player
            
        Returns:
            Boolean array of size action_space.n
        """
        pass

