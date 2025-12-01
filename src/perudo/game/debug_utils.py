"""
Debug utilities for Perudo vectorized environment.

Provides debug logging functions for player moves and game state.
"""

from typing import Optional, Any
from ..utils.helpers import action_to_bid


def log_player_move(
    action: int,
    player_id: int,
    env: Any,
    max_quantity: int,
    player_name: Optional[str] = None,
    stage: str = "AFTER",
    reward: Optional[float] = None,
    done: Optional[bool] = None,
) -> None:
    """
    Log player move in debug mode.
    
    Args:
        action: Action taken
        player_id: Player ID
        env: Environment instance
        max_quantity: Maximum dice quantity
        player_name: Custom player name (if None, uses "Player {player_id}")
        stage: Stage identifier (e.g., "BEFORE", "AFTER")
        reward: Reward received (optional, only shown if provided)
        done: Whether episode is done (optional, only shown if provided)
    """
    try:
        action_type, param1, param2 = action_to_bid(action, max_quantity)
        
        if player_name is None:
            player_name = f"Player {player_id}"
        
        if action_type == "bid":
            move_str = f"{player_name}: BID {param1}x{param2}"
        elif action_type == "challenge":
            move_str = f"{player_name}: CHALLENGE"
        elif action_type == "believe":
            move_str = f"{player_name}: BELIEVE"
        else:
            move_str = f"{player_name}: {action_type}"
        
        # Show current game state
        current_bid = env.game_state.current_bid
        bid_str = f"{current_bid[0]}x{current_bid[1]}" if current_bid else "none"
        dice_str = str(list(env.game_state.player_dice_count))
        
        # Build log message
        log_parts = [f"[DEBUG] {move_str} | Current bid: {bid_str} | Dice: {dice_str}"]
        
        if reward is not None:
            log_parts.append(f"Reward: {reward:.2f}")
        
        if stage:
            log_parts.append(f"{stage} step")
        
        if done is not None:
            log_parts.append(f"Done: {done}, Current player: {env.game_state.current_player}")
            # Add learning agent dice count if it's a learning agent
            if player_id == 0:
                learning_agent_dice = env.game_state.player_dice_count[0]
                log_parts.append(f"Learning agent dice: {learning_agent_dice}")
        
        print(" | ".join(log_parts))
        
        # Pause and wait for Enter key
        wait_for_debug_input()
    except Exception as e:
        # Print error instead of silently ignoring
        print(f"[DEBUG] Error in debug logging: {e}")
        import traceback
        traceback.print_exc()
        # Try to pause even if there was an error
        try:
            wait_for_debug_input("Press Enter to continue (after error)...")
        except:
            pass


def log_simple_message(message: str) -> None:
    """
    Log a simple debug message.
    
    Args:
        message: Message to log
    """
    print(f"[DEBUG] {message}")


def wait_for_debug_input(message: str = "Press Enter to continue...") -> None:
    """
    Wait for user input in debug mode.
    
    Args:
        message: Message to display before waiting
    """
    try:
        print(message)
        input()
    except (EOFError, KeyboardInterrupt):
        # Handle case where stdin is not available or user cancels
        pass
    except Exception as e:
        # Silently ignore other errors to prevent crashes
        print(f"[DEBUG] Input error: {e}")


def format_game_state(env: Any) -> str:
    """
    Format current game state as a string.
    
    Args:
        env: Environment instance
        
    Returns:
        Formatted game state string
    """
    current_bid = env.game_state.current_bid
    bid_str = f"{current_bid[0]}x{current_bid[1]}" if current_bid else "none"
    dice_str = str(list(env.game_state.player_dice_count))
    current_player = env.game_state.current_player
    game_over = env.game_state.game_over
    
    return (
        f"Current bid: {bid_str} | "
        f"Dice: {dice_str} | "
        f"Current player: {current_player} | "
        f"Game over: {game_over}"
    )







