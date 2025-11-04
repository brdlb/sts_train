"""
Helper functions for working with Perudo.
"""

from typing import List, Tuple, Optional
import numpy as np


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
    - 1: pacao
    - 2+: bids (quantity, value) encoded as encode_bid(quantity, value)

    Args:
        max_players: Maximum number of players
        max_quantity: Maximum dice quantity in bid

    Returns:
        Action space size
    """
    # 2 special actions (challenge, pacao) + all possible bids
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
        Types: 'challenge', 'pacao', 'bid'
    """
    if action == 0:
        return ("challenge", None, None)
    elif action == 1:
        return ("pacao", None, None)
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
    return 2 + encoded  # Offset by 2 (challenge and pacao)


def create_observation_vector(
    current_bid: Optional[Tuple[int, int]],
    bid_history: List[Tuple[int, int, int]],
    player_dice_count: List[int],
    current_player: int,
    palifico_active: List[bool],
    pacao_called: bool,
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
        pacao_called: Pacao call flag
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

    # Pacao flag (1 value)
    obs_parts.append([1 if pacao_called else 0])

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
    pacao_success: Optional[bool] = None,
    dice_lost: int = 0,
) -> float:
    """
    Calculate reward for agent.

    Args:
        action_type: Action type ('bid', 'challenge', 'pacao')
        game_over: Whether game is over
        winner: Winner ID (if game is over)
        player_id: Current player ID
        challenge_success: Whether challenge succeeded (if applicable)
        pacao_success: Whether pacao succeeded (if applicable)
        dice_lost: Number of dice lost

    Returns:
        Reward
    """
    reward = 0.0

    # Reward for winning game
    if game_over and winner == player_id:
        reward += 100.0

    # Penalty for losing dice
    if dice_lost > 0:
        reward -= 10.0 * dice_lost

    # Intermediate rewards for bluffs and challenges
    # +0.5 for successful bluff or catching someone else's bluff
    # -0.5 for unsuccessful bet or challenge that led to losing a die
    if action_type == "challenge" and challenge_success is not None:
        if challenge_success:
            # Successfully caught someone's bluff
            reward += 0.5
        else:
            # Unsuccessful challenge that led to dice loss
            if dice_lost > 0:
                reward -= 0.5

    if action_type == "pacao" and pacao_success is not None:
        if pacao_success:
            # Successfully called pacao (caught someone's bluff)
            reward += 0.5
        else:
            # Unsuccessful pacao that led to dice loss
            if dice_lost > 0:
                reward -= 0.5

    # Note: Successful bluff detection (bid that was never challenged) 
    # is handled separately in the environment when round ends

    return reward
