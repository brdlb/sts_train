"""
Debug logger for game debugging.

This module provides centralized debug logging functionality.
"""

from typing import Optional, Dict, Any
from .interfaces import IGameController
from ..training.config import DebugConfig, DEFAULT_CONFIG
from ..utils.helpers import action_to_bid


class DebugLogger:
    """
    Centralized debug logging.
    
    Responsibilities:
    - Log player actions
    - Display game state
    - Control debug pauses
    
    Contract:
    - Can be completely disabled (no-op mode)
    - Does not affect game logic
    - Thread-safe
    """
    
    def __init__(self, config: Optional[DebugConfig] = None):
        """
        Initialize debug logger.
        
        Args:
            config: Debug configuration (uses DEFAULT_CONFIG if not provided)
        """
        self.config = config if config is not None else DEFAULT_CONFIG.environment.debug
    
    def log_action(
        self, 
        player_id: int, 
        action_type: str, 
        params: Dict[str, Any], 
        game_state,
        max_quantity: int = 30
    ) -> None:
        """
        Log player action.
        
        Args:
            player_id: ID of player
            action_type: Type of action ('bid', 'challenge', 'believe')
            params: Action parameters (quantity, value for bids)
            game_state: Current game state
            max_quantity: Maximum quantity for decoding actions
        """
        if not self.config.enabled or not self.config.log_actions:
            return
        
        player_name = f"Player {player_id}"
        
        if action_type == "bid":
            quantity = params.get("quantity", params.get("param1"))
            value = params.get("value", params.get("param2"))
            move_str = f"{player_name}: BID {quantity}x{value}"
        elif action_type == "challenge":
            move_str = f"{player_name}: CHALLENGE"
        elif action_type == "believe":
            move_str = f"{player_name}: BELIEVE"
        else:
            move_str = f"{player_name}: {action_type}"
        
        # Show current game state
        current_bid = game_state.current_bid
        bid_str = f"{current_bid[0]}x{current_bid[1]}" if current_bid else "none"
        dice_str = str(list(game_state.player_dice_count))
        
        print(f"[DEBUG] {move_str} | Current bid: {bid_str} | Dice: {dice_str}")
    
    def log_game_state(self, game_state, prefix: str = "[DEBUG]") -> None:
        """
        Log game state.
        
        Args:
            game_state: Current game state
            prefix: Prefix for log message
        """
        if not self.config.enabled or not self.config.log_game_state:
            return
        
        print(f"{prefix} Game State:")
        print(f"  Current player: {game_state.current_player}")
        if game_state.current_bid:
            q, v = game_state.current_bid
            print(f"  Current bid: {q}x{v}")
        else:
            print("  Current bid: none")
        print(f"  Player dice counts: {game_state.player_dice_count}")
        print(f"  Palifico active: {game_state.palifico_active}")
        print(f"  Game over: {game_state.game_over}")
        if game_state.winner is not None:
            print(f"  Winner: Player {game_state.winner}")
    
    def wait_for_input(self) -> None:
        """Wait for user input (for debugging)."""
        if not self.config.enabled or not self.config.wait_for_input:
            return
        
        try:
            input("Press Enter to continue...")
        except (EOFError, KeyboardInterrupt):
            # Handle case where stdin is not available or user cancels
            pass
    
    def print_episode_summary(
        self, 
        winner: int, 
        reward: float, 
        stats: Dict[str, int], 
        dice_counts: list
    ) -> None:
        """
        Print episode summary.
        
        Args:
            winner: Winner ID
            reward: Final reward
            stats: Episode statistics (bid_count, challenge_count, etc.)
            dice_counts: Final dice counts for all players
        """
        if not self.config.enabled:
            return
        
        learning_agent_won = winner == 0
        win_status = "WIN" if learning_agent_won else "DEFEAT"
        dice_str = str(list(dice_counts))
        
        if self.config.verbose_episode_summary:
            print(f"{win_status} | reward: {reward:.2f} | "
                  f"stats: bids={stats.get('bid_count', 0)}, "
                  f"challenges={stats.get('challenge_count', 0)}, "
                  f"believe={stats.get('believe_count', 0)}, "
                  f"invalid={stats.get('invalid_action_count', 0)} | "
                  f"dice: {dice_str}")
        else:
            print(f"{win_status} | reward: {reward:.2f} | dice: {dice_str}")

