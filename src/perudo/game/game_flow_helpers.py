"""
Game flow helper functions for Perudo vectorized environment.

Handles game flow logic like advancing to learning agent, action validation,
and observation retrieval.
"""

from typing import Tuple, Optional, Any, Callable, List, Dict
from ..utils.helpers import action_to_bid, bid_to_action


def validate_and_fix_action(
    action: int,
    env: Any,
    player_id: int,
    max_quantity: int,
) -> int:
    """
    Validate and fix action if current_bid is None.
    
    CRITICAL: If current_bid is None, player MUST make a bid, not challenge or believe.
    This function checks if action is challenge/believe when there's no bid and forces a bid instead.
    
    Args:
        action: Action to validate
        env: Environment instance
        player_id: Player ID making the action
        max_quantity: Maximum dice quantity
        
    Returns:
        Validated (and possibly fixed) action
    """
    current_bid = env.game_state.current_bid
    if current_bid is None:
        # No bid yet - must make initial bid, not challenge or believe
        action_type, _, _ = action_to_bid(action, max_quantity)
        if action_type == "challenge" or action_type == "believe":
            # Force player to make initial bid instead
            # Get player dice to make a reasonable initial bid
            player_dice = env.game_state.get_player_dice(player_id)
            if player_dice:
                # Make minimal initial bid
                # Use first die value, but not 1 (wildcard), use 2 instead
                face = player_dice[0] if player_dice[0] != 1 else 2
                action = bid_to_action(1, face, max_quantity)
            else:
                # Fallback: make minimal safe bid
                action = bid_to_action(1, 2, max_quantity)
    
    return action


def get_observation_for_agent(
    env: Any,
    agent_id: int,
    all_agents_learn_mode: bool,
    active_agent_id: int,
) -> Any:
    """
    Get observation for an agent.
    
    In all_learn mode, use current active agent.
    In normal mode, use agent_id (usually 0 for learning agent).
    
    Args:
        env: Environment instance
        agent_id: Agent ID (usually 0 for learning agent)
        all_agents_learn_mode: Whether all agents are learning
        active_agent_id: Current active agent ID
        
    Returns:
        Observation for the agent
    """
    if all_agents_learn_mode:
        return env.get_observation_for_player(active_agent_id)
    else:
        return env.get_observation_for_player(agent_id)


def advance_to_learning_agent(
    env: Any,
    active_agent_id: int,
    opponent_manager: Any,
    env_idx: int,
    all_agents_learn_mode: bool,
    max_steps: int = 100,
    debug_mode: Optional[Any] = None,
    seeds: Optional[int] = None,
    options: Optional[Dict] = None,
    on_reset_callback: Optional[Callable[[int, Any], Tuple[Any, Any]]] = None,
    on_opponent_action_callback: Optional[Callable[[int, int, int, Any], None]] = None,
) -> Tuple[int, bool, Optional[Any], Optional[Any]]:
    """
    Advance game to learning agent's turn, executing opponent moves.
    
    This function skips opponent turns until it's the learning agent's turn.
    Handles edge cases like:
    - Players with no dice (eliminated)
    - Game ending before learning agent's turn (requires reset)
    - Debug logging of opponent moves
    
    Args:
        env: Environment instance
        active_agent_id: Current active agent ID (will be updated)
        opponent_manager: OpponentManager instance
        env_idx: Environment index
        all_agents_learn_mode: Whether all agents are learning (if True, doesn't skip)
        max_steps: Maximum number of steps to take (safety limit)
        debug_mode: Debug mode flag (optional)
        seeds: Seed for reset (optional)
        options: Options for reset (optional)
        on_reset_callback: Callback function called when reset is needed
                          Signature: (env_idx, env) -> (obs, info)
        on_opponent_action_callback: Callback function called after opponent action
                                    Signature: (env_idx, opponent_idx, player_id, action) -> None
    
    Returns:
        Tuple of (new_active_agent_id, game_over, obs, info)
        - new_active_agent_id: Updated active agent ID
        - game_over: Whether game is over
        - obs: Observation after advancement (None if reset happened)
        - info: Info dict after advancement (None if reset happened)
    """
    # Skip only if not in all_learn mode (in all_learn mode, all agents are learning)
    if all_agents_learn_mode:
        # All agents learn, so current agent is already a learning agent
        return active_agent_id, False, None, None
    
    steps = 0
    actual_num_players = env.num_players
    
    while active_agent_id != 0 and steps < max_steps:
        steps += 1
        
        # CRITICAL: Skip players with no dice - they have already lost and cannot make moves
        # Use next_player() which automatically skips players with 0 dice
        if env.game_state.player_dice_count[active_agent_id] == 0:
            env.game_state.next_player()
            active_agent_id = env.game_state.current_player
            # Check if game ended after skipping players
            if env.game_state.game_over:
                # Game ended, reset again
                if on_reset_callback is not None:
                    obs, info = on_reset_callback(env_idx, env)
                else:
                    obs, info = env.reset(seed=seeds, options=options)
                active_agent_id = env.game_state.current_player
                steps = 0  # Start over
                continue
            continue
        
        # Opponent's turn
        opponent_idx = active_agent_id - 1
        # Only use opponent if opponent_idx is within bounds
        if opponent_idx < len(opponent_manager.opponent_models[env_idx]) or (
            opponent_manager.use_bot_opponents and 
            opponent_idx < len(opponent_manager.opponent_bots[env_idx])
        ):
            obs_for_opp = env.get_observation_for_player(active_agent_id)
            action = opponent_manager.get_opponent_action(
                env_idx, opponent_idx, active_agent_id, obs_for_opp
            )
            
            # Validate and fix action (must make bid if current_bid is None)
            action = validate_and_fix_action(action, env, active_agent_id, env.max_quantity)
            
            # Call callback if provided (for debug logging)
            if on_opponent_action_callback is not None:
                on_opponent_action_callback(env_idx, opponent_idx, active_agent_id, action)
            
            env.set_active_player(active_agent_id)
            obs_for_opp, _, terminated, opp_truncated, opp_info = env.step(action)
            
            # Debug mode: log opponent's move
            if debug_mode is not None and debug_mode.is_set():
                _log_player_move_debug(action, active_agent_id, env, env.max_quantity)
            
            if terminated or opp_truncated:
                # Game ended before learning agent's turn, reset again
                if on_reset_callback is not None:
                    obs, info = on_reset_callback(env_idx, env)
                else:
                    obs, info = env.reset(seed=seeds, options=options)
                active_agent_id = env.game_state.current_player
                steps = 0  # Start over
                continue
            
            # Update active agent ID
            active_agent_id = env.game_state.current_player
        else:
            # No opponent available, break out
            break
    
    # Check if game is over
    game_over = env.game_state.game_over
    
    return active_agent_id, game_over, None, None


def _log_player_move_debug(action: int, player_id: int, env: Any, max_quantity: int) -> None:
    """
    Log player move in debug mode (internal helper).
    
    Args:
        action: Action taken
        player_id: Player ID
        env: Environment instance
        max_quantity: Maximum dice quantity
    """
    try:
        action_type, param1, param2 = action_to_bid(action, max_quantity)
        
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
        
        print(f"[DEBUG] {move_str} | Current bid: {bid_str} | Dice: {dice_str}")
        # Pause and wait for Enter key
        try:
            input("Press Enter to continue...")
        except (EOFError, KeyboardInterrupt):
            # Handle case where stdin is not available or user cancels
            pass
    except Exception:
        # Silently ignore errors to prevent crashes
        pass


def handle_game_termination(
    env: Any,
    winner: int,
    episode_stats: dict,
) -> dict:
    """
    Handle game termination and create episode info.
    
    Args:
        env: Environment instance
        winner: Winner player ID
        episode_stats: Episode statistics dictionary
        
    Returns:
        Episode info dictionary
    """
    info = {
        "game_over": True,
        "winner": winner,
        "episode": episode_stats,
        "episode_reward": episode_stats.get("r", 0.0),
        "episode_length": episode_stats.get("l", 0),
    }
    return info

