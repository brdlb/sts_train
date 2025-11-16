"""
Helper functions for working with Perudo.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
from ..training.config import RewardConfig, DEFAULT_CONFIG


def encode_bid(quantity: int, value: int, max_quantity: int = 30) -> int:
    """
    Encode bid into a single number.

    Args:
        quantity: Number of dice
        value: Dice value (1-6)
        max_quantity: Maximum quantity for encoding

    Returns:
        Encoded number
    """
    return (quantity - 1) * 6 + (value - 1)


def decode_bid(encoded: int, max_quantity: int = 30) -> Tuple[int, int]:
    """
    Decode bid from a single number.

    Args:
        encoded: Encoded number
        max_quantity: Maximum quantity for decoding

    Returns:
        Tuple (quantity, value)
    """
    quantity = (encoded // 6) + 1
    value = (encoded % 6) + 1
    return quantity, value


def get_action_space_size(max_players: int = 6, max_quantity: int = 30) -> int:
    """
    Get action space size.

    Actions:
    - 0: challenge
    - 1: believe
    - 2+: bids (quantity, value) encoded as encode_bid(quantity, value)

    Args:
        max_players: Maximum number of players
        max_quantity: Maximum dice quantity in bid

    Returns:
        Action space size
    """
    # 2 special actions (challenge, believe) + all possible bids
    bid_actions = max_quantity * 6  # Maximum 30 * 6 = 180 bids
    return 2 + bid_actions


def action_to_bid(action: int, max_quantity: int = 30) -> Tuple[str, Optional[int], Optional[int]]:
    """
    Convert action to format (type, param1, param2).

    Args:
        action: Action from action_space
        max_quantity: Maximum dice quantity

    Returns:
        Tuple (action_type, param1, param2)
        Types: 'challenge', 'believe', 'bid'
    """
    if action == 0:
        return ("challenge", None, None)
    elif action == 1:
        return ("believe", None, None)
    else:
        # Actions 2+ are bids
        bid_encoded = action - 2
        quantity, value = decode_bid(bid_encoded, max_quantity)
        return ("bid", quantity, value)


def bid_to_action(quantity: int, value: int, max_quantity: int = 30) -> int:
    """
    Convert bid to action.

    Args:
        quantity: Number of dice
        value: Dice value
        max_quantity: Maximum dice quantity

    Returns:
        Action in action_space
    """
    encoded = encode_bid(quantity, value, max_quantity)
    return 2 + encoded  # Offset by 2 (challenge and believe)


def create_observation_dict(
    current_bid: Optional[Tuple[int, int]],
    bid_history: List[Tuple[int, int, int]],
    player_dice_count: List[int],
    current_player: int,
    palifico_active: List[bool],
    believe_called: bool,
    player_dice: List[int],  # Current player's dice (visible only to them)
    max_history_length: int = 20,
    max_players: int = 6,
    agent_id: Optional[int] = None,
    num_agents: int = 4,
    special_round_active: bool = False,
    round_number: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Create observation dictionary for transformer-based agent.
    
    Args:
        current_bid: Current bid (quantity, value)
        bid_history: Bid history as list of (player_id, quantity, value)
        player_dice_count: Number of dice for each player
        current_player: Current player
        palifico_active: Palifico flags for each player
        believe_called: Believe call flag
        player_dice: Current player's dice
        max_history_length: Maximum length of bid history sequence
        max_players: Maximum number of players
        agent_id: Agent ID for one-hot encoding (0 to num_agents-1)
        num_agents: Total number of agents (for one-hot encoding)
        special_round_active: Whether special round is active
        round_number: Current round number
    
    Returns:
        Dictionary with 'bid_history' and 'static_info' keys
        bid_history shape: (max_history_length, 3) - (player_id, quantity, value)
    """
    # Build bid_history sequence (max_history_length, 3) - player_id, quantity, value
    # This preserves information about who made each bid, which is crucial for understanding
    # player strategies and turn order context
    bid_history_array = np.zeros((max_history_length, 3), dtype=np.int32)
    for i in range(max_history_length):
        # Take from end of history (most recent first)
        history_idx = len(bid_history) - 1 - i
        if history_idx >= 0:
            player_id, quantity, value = bid_history[history_idx]
            bid_history_array[i] = [player_id, quantity, value]
        # else: padding (already zeros)
    
    # Build static_info vector
    static_parts = []
    
    # Agent ID one-hot encoding (num_agents values)
    if agent_id is not None:
        agent_id_onehot = np.zeros(num_agents, dtype=np.float32)
        if 0 <= agent_id < num_agents:
            agent_id_onehot[agent_id] = 1.0
        static_parts.append(agent_id_onehot)
    else:
        static_parts.append(np.zeros(num_agents, dtype=np.float32))
    
    # Current bid (2 values: quantity, value) or (0, 0) if no bid
    if current_bid is not None:
        static_parts.append(np.array([current_bid[0], current_bid[1]], dtype=np.float32))
    else:
        static_parts.append(np.array([0.0, 0.0], dtype=np.float32))
    
    # Number of dice for each player (max_players values)
    dice_count_padded = player_dice_count + [0] * (max_players - len(player_dice_count))
    static_parts.append(np.array(dice_count_padded[:max_players], dtype=np.float32))
    
    # Current player (1 value)
    static_parts.append(np.array([current_player], dtype=np.float32))
    
    # Palifico flags (max_players values)
    palifico_padded = [1.0 if p else 0.0 for p in palifico_active] + [0.0] * (
        max_players - len(palifico_active)
    )
    static_parts.append(np.array(palifico_padded[:max_players], dtype=np.float32))
    
    # Believe flag (1 value)
    static_parts.append(np.array([1.0 if believe_called else 0.0], dtype=np.float32))
    
    # Special round active flag (1 value)
    static_parts.append(np.array([1.0 if special_round_active else 0.0], dtype=np.float32))
    
    # Round number (1 value)
    static_parts.append(np.array([float(round_number)], dtype=np.float32))
    
    # Current player's dice (5 values, pad with zeros if less)
    dice_padded = player_dice + [0] * (5 - len(player_dice))
    static_parts.append(np.array(dice_padded[:5], dtype=np.float32))
    
    # Combine static parts
    static_info = np.concatenate(static_parts, dtype=np.float32)
    
    return {
        "bid_history": bid_history_array,
        "static_info": static_info,
    }


def create_observation_vector(
    current_bid: Optional[Tuple[int, int]],
    bid_history: List[Tuple[int, int, int]],
    player_dice_count: List[int],
    current_player: int,
    palifico_active: List[bool],
    believe_called: bool,
    player_dice: List[int],  # Current player's dice (visible only to them)
    history_length: int = 10,
    max_players: int = 6,
    agent_id: Optional[int] = None,
    num_agents: int = 4,
) -> np.ndarray:
    """
    Create observation vector for agent.

    Args:
        current_bid: Current bid (quantity, value)
        bid_history: Bid history
        player_dice_count: Number of dice for each player
        current_player: Current player
        palifico_active: Palifico flags for each player
        believe_called: Believe call flag
        player_dice: Current player's dice
        history_length: Bid history length
        max_players: Maximum number of players
        agent_id: Agent ID for one-hot encoding (0 to num_agents-1)
        num_agents: Total number of agents (for one-hot encoding)

    Returns:
        Observation vector
    """
    obs_parts = []

    # Agent ID one-hot encoding (num_agents values)
    if agent_id is not None:
        agent_id_onehot = np.zeros(num_agents, dtype=np.float32)
        if 0 <= agent_id < num_agents:
            agent_id_onehot[agent_id] = 1.0
        obs_parts.append(agent_id_onehot)
    else:
        # If no agent_id provided, use zeros
        obs_parts.append(np.zeros(num_agents, dtype=np.float32))

    # Current bid (2 values: quantity, value) or (0, 0) if no bid
    if current_bid is not None:
        obs_parts.append([current_bid[0], current_bid[1]])
    else:
        obs_parts.append([0, 0])

    # Bid history (last N bids, each = 3 values: player, quantity, value)
    history_vector = []
    for i in range(history_length):
        if i < len(bid_history):
            player_id, quantity, value = bid_history[-(i + 1)]  # Take from end
            history_vector.extend([player_id, quantity, value])
        else:
            history_vector.extend([0, 0, 0])
    obs_parts.append(history_vector)

    # Number of dice for each player (max_players values)
    dice_count_padded = player_dice_count + [0] * (max_players - len(player_dice_count))
    obs_parts.append(dice_count_padded[:max_players])

    # Current player (1 value)
    obs_parts.append([current_player])

    # Palifico flags (max_players values)
    palifico_padded = [1 if p else 0 for p in palifico_active] + [0] * (
        max_players - len(palifico_active)
    )
    obs_parts.append(palifico_padded[:max_players])

    # Believe flag (1 value)
    obs_parts.append([1 if believe_called else 0])

    # Current player's dice (5 values, pad with zeros if less)
    dice_padded = player_dice + [0] * (5 - len(player_dice))
    obs_parts.append(dice_padded[:5])

    # Combine all parts
    obs = np.concatenate(obs_parts, dtype=np.float32)
    return obs


def calculate_reward(
    action_type: str,
    game_over: bool,
    winner: int,
    player_id: int,
    challenge_success: Optional[bool] = None,
    believe_success: Optional[bool] = None,
    dice_lost: int = 0,
    reward_config: Optional[RewardConfig] = None,
    winner_dice_count: Optional[int] = None,
) -> float:
    """
    Calculate intermediate step reward for agent action.

    This function calculates ONLY intermediate step rewards (bid penalties, challenge rewards, etc.).
    Final episode rewards (win_reward, win_dice_bonus, lose_penalty, dice_lost_penalty) are
    calculated separately in RewardManager.calculate_final_reward() to avoid double counting.

    Args:
        action_type: Action type ('bid', 'challenge', 'believe')
        game_over: Whether game is over (not used for final rewards, only for documentation)
        winner: Winner ID (not used for final rewards, only for documentation)
        player_id: Current player ID
        challenge_success: Whether challenge succeeded (if applicable)
        believe_success: Whether believe succeeded (if applicable)
        dice_lost: Number of dice lost (used for challenge/believe failure penalties only)
        reward_config: Reward configuration (uses DEFAULT_CONFIG if not provided)
        winner_dice_count: Number of dice remaining for winner (deprecated, not used)

    Returns:
        Intermediate step reward (does not include final episode rewards)
    """
    if reward_config is None:
        reward_config = DEFAULT_CONFIG.reward

    reward = 0.0

    # NOTE: Final episode rewards (win_reward, win_dice_bonus, lose_penalty) are NOT applied here.
    # They are applied only once at episode end in RewardManager.calculate_final_reward()
    # to avoid double counting. This function only calculates intermediate step rewards.

    # NOTE: Penalty for losing dice (dice_lost_penalty) is NOT applied here - it's applied only at episode end
    # in RewardManager.calculate_final_reward() to avoid double counting.
    # The dice_lost parameter is still used for challenge/believe failure penalties below,
    # but NOT for the per-die penalty which is calculated separately at episode end.

    # Small penalty for bidding to encourage finishing the round
    if action_type == "bid":
        reward += reward_config.bid_small_penalty

    # Intermediate rewards for bluffs and challenges
    if action_type == "challenge" and challenge_success is not None:
        if challenge_success:
            # Successfully caught someone's bluff
            reward += reward_config.challenge_success_reward
        else:
            # Unsuccessful challenge that led to dice loss
            if dice_lost > 0:
                reward += reward_config.challenge_failure_penalty

    if action_type == "believe" and believe_success is not None:
        if believe_success:
            # Successfully called believe (caught someone's bluff)
            reward += reward_config.believe_success_reward
        else:
            # Unsuccessful believe that led to dice loss
            if dice_lost > 0:
                reward += reward_config.believe_failure_penalty

    # Note: Successful bluff detection (bid that was never challenged) 
    # is handled separately in the environment when round ends

    return reward


def create_action_mask(
    available_actions: List[Tuple[str, Optional[int], Optional[int]]],
    action_space_size: int,
    max_quantity: int = 30,
) -> np.ndarray:
    """
    Create a boolean mask for available actions.

    Args:
        available_actions: List of available actions from PerudoRules.get_available_actions
        action_space_size: Total size of the action space
        max_quantity: Maximum dice quantity in bid

    Returns:
        Boolean numpy array where True means the action is available.
    """
    mask = np.zeros(action_space_size, dtype=bool)
    for action_type, param1, param2 in available_actions:
        if action_type == "challenge":
            mask[0] = True
        elif action_type == "believe":
            mask[1] = True
        elif action_type == "bid":
            quantity, value = param1, param2
            action_idx = bid_to_action(quantity, value, max_quantity)
            if 0 <= action_idx < action_space_size:
                mask[action_idx] = True
    return mask
