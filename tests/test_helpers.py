"""
Tests for helper functions.
"""

import pytest
import numpy as np
from src.perudo.utils.helpers import (
    create_observation_vector,
    encode_bid,
    decode_bid,
    action_to_bid,
    bid_to_action,
)


def test_create_observation_vector_with_agent_id():
    """Test creating observation vector with agent_id."""
    # Create observation for agent 0
    obs_0 = create_observation_vector(
        current_bid=None,
        bid_history=[],
        player_dice_count=[5, 5, 5, 5],
        current_player=0,
        palifico_active=[False, False, False, False],
        believe_called=False,
        player_dice=[1, 2, 3, 4, 5],
        history_length=10,
        max_players=4,
        agent_id=0,
        num_agents=4,
    )
    
    # Create observation for agent 1
    obs_1 = create_observation_vector(
        current_bid=None,
        bid_history=[],
        player_dice_count=[5, 5, 5, 5],
        current_player=0,
        palifico_active=[False, False, False, False],
        believe_called=False,
        player_dice=[1, 2, 3, 4, 5],
        history_length=10,
        max_players=4,
        agent_id=1,
        num_agents=4,
    )
    
    # Check agent_id one-hot encoding
    assert obs_0[0] == 1.0  # Agent 0
    assert obs_0[1] == 0.0
    assert obs_0[2] == 0.0
    assert obs_0[3] == 0.0
    
    assert obs_1[0] == 0.0
    assert obs_1[1] == 1.0  # Agent 1
    assert obs_1[2] == 0.0
    assert obs_1[3] == 0.0
    
    # Check that observations have the same shape
    assert obs_0.shape == obs_1.shape


def test_create_observation_vector_without_agent_id():
    """Test creating observation vector without agent_id."""
    obs = create_observation_vector(
        current_bid=None,
        bid_history=[],
        player_dice_count=[5, 5],
        current_player=0,
        palifico_active=[False, False],
        believe_called=False,
        player_dice=[1, 2, 3, 4, 5],
        history_length=10,
        max_players=2,
        agent_id=None,
        num_agents=4,
    )
    
    # Agent ID should be zeros if not provided
    assert obs[0] == 0.0
    assert obs[1] == 0.0
    assert obs[2] == 0.0
    assert obs[3] == 0.0


def test_encode_decode_bid():
    """Test bid encoding and decoding."""
    quantity, value = 5, 3
    
    encoded = encode_bid(quantity, value)
    decoded_quantity, decoded_value = decode_bid(encoded)
    
    assert decoded_quantity == quantity
    assert decoded_value == value


def test_action_to_bid():
    """Test converting action to bid."""
    # Challenge action
    action_type, param1, param2 = action_to_bid(0, max_quantity=30)
    assert action_type == "challenge"
    assert param1 is None
    assert param2 is None
    
    # Believe action
    action_type, param1, param2 = action_to_bid(1, max_quantity=30)
    assert action_type == "believe"
    assert param1 is None
    assert param2 is None
    
    # Bid action
    action_type, param1, param2 = action_to_bid(2, max_quantity=30)
    assert action_type == "bid"
    assert param1 is not None
    assert param2 is not None


def test_bid_to_action():
    """Test converting bid to action."""
    quantity, value = 5, 3
    
    action = bid_to_action(quantity, value, max_quantity=30)
    
    # Should be at least 2 (offset for challenge and believe)
    assert action >= 2
    
    # Decode back
    action_type, param1, param2 = action_to_bid(action, max_quantity=30)
    assert action_type == "bid"
    assert param1 == quantity
    assert param2 == value


def test_create_observation_vector_shape():
    """Test observation vector shape with agent_id."""
    obs = create_observation_vector(
        current_bid=None,
        bid_history=[],
        player_dice_count=[5, 5, 5, 5],
        current_player=0,
        palifico_active=[False, False, False, False],
        believe_called=False,
        player_dice=[1, 2, 3, 4, 5],
        history_length=10,
        max_players=4,
        agent_id=0,
        num_agents=4,
    )
    
    # Expected size: agent_id(4) + current_bid(2) + history(10*3) + 
    # dice_count(4) + current_player(1) + palifico(4) + believe(1) + player_dice(5)
    expected_size = 4 + 2 + 10 * 3 + 4 + 1 + 4 + 1 + 5
    assert obs.shape == (expected_size,)
    assert obs.dtype == np.float32

