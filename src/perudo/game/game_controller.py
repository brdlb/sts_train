"""
Game controller for managing game process.

This module handles:
- Skipping players without dice
- Handling round end
- Forcing initial bids when needed
"""

from typing import Optional, Tuple
from .interfaces import IGameController
from ..training.config import GameControllerConfig, DEFAULT_CONFIG
from ..utils.helpers import action_to_bid, bid_to_action


class GameController(IGameController):
    """
    Controller for managing game process.
    
    Responsibilities:
    - Skip players without dice
    - Handle round end (reset bid, roll dice)
    - Force initial bids (replace challenge/believe with bid at round start)
    - Check game continuation conditions
    
    Contract:
    - Methods modify game_state only through public GameState methods
    - All methods are thread-safe (no mutable state)
    - Returns new value or None if operation failed
    """
    
    def __init__(self, config: Optional[GameControllerConfig] = None):
        """
        Initialize game controller.
        
        Args:
            config: Game controller configuration (uses DEFAULT_CONFIG if not provided)
        """
        self.config = config if config is not None else DEFAULT_CONFIG.environment.game_controller
    
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
        num_players = game_state.num_players
        attempts = 0
        
        # Start from next player
        next_player = (current_player + 1) % num_players
        
        while attempts < self.config.max_skipped_players:
            # Check if this player has dice
            if game_state.player_dice_count[next_player] > 0:
                return next_player
            
            # Move to next player
            next_player = (next_player + 1) % num_players
            attempts += 1
        
        # No active players found
        raise ValueError(f"No active players found after {self.config.max_skipped_players} attempts")
    
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
        # Clear bid state
        game_state.current_bid = None
        game_state.believe_called = False
        
        # Reset special round at end of round
        game_state.special_round_active = False
        game_state.special_round_declared_by = None
        
        # Set next round starter
        starter_id = winner_id if winner_id is not None else loser_id
        
        # If starter has no dice (eliminated), skip to next active player
        if game_state.player_dice_count[starter_id] == 0:
            try:
                starter_id = self.skip_to_next_active_player(game_state, starter_id)
            except ValueError:
                # No active players - game should be over
                game_state._check_game_over()
                return
        
        game_state.current_player = starter_id
        
        # Roll dice again - round ends
        game_state.roll_dice()
    
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
        # Check if we need to force initial bid
        if not self.config.auto_force_initial_bid:
            return action, False
        
        # If there's already a bid, no need to force
        if game_state.current_bid is not None:
            return action, False
        
        # Check if action is challenge or believe
        action_type, _, _ = action_to_bid(action, max_quantity)
        
        if action_type not in ("challenge", "believe"):
            return action, False
        
        # Force initial bid
        # Try to use player's dice to make reasonable bid
        player_dice = game_state.get_player_dice(player_id)
        
        if player_dice:
            # Use first die value (but not 1, as 1s are jokers in normal round)
            face = player_dice[0] if player_dice[0] != 1 else self.config.fallback_bid_value
            new_action = bid_to_action(
                self.config.fallback_bid_quantity, 
                face, 
                max_quantity
            )
        else:
            # Fallback: use default bid
            new_action = bid_to_action(
                self.config.fallback_bid_quantity,
                self.config.fallback_bid_value,
                max_quantity
            )
        
        return new_action, True

