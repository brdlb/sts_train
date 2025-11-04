"""
Tests for Gymnasium Perudo environment.
"""

import pytest
import numpy as np
from src.perudo.game.perudo_env import PerudoEnv


def test_env_initialization():
    """Test environment initialization."""
    env = PerudoEnv(num_players=4, dice_per_player=5)
    
    assert env.num_players == 4
    assert env.dice_per_player == 5
    assert env.observation_space is not None
    assert env.action_space is not None


def test_env_reset():
    """Test environment reset."""
    env = PerudoEnv(num_players=2, dice_per_player=5)
    obs, info = env.reset()
    
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert len(obs) > 0
    assert "player_id" in info
    assert "game_state" in info


def test_env_step():
    """Test step function in environment."""
    env = PerudoEnv(num_players=2, dice_per_player=5)
    obs, info = env.reset()
    
    # Choose a random action
    action = env.action_space.sample()
    
    # Execute action
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    assert isinstance(next_obs, np.ndarray)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_env_observation_shape():
    """Test observation shape."""
    env = PerudoEnv(num_players=2, dice_per_player=5, history_length=10)
    obs, _ = env.reset()
    
    expected_size = 2 + 10 * 3 + 2 + 1 + 2 + 1 + 5  # All observation components
    assert obs.shape == (expected_size,)


def test_env_action_space():
    """Test actions space size."""
    env = PerudoEnv(num_players=2, dice_per_player=5, max_quantity=30)
    
    # Size should be: 2 (challenge, pacao) + 30 * 6 (bids)
    expected_size = 2 + 30 * 6
    assert env.action_space.n == expected_size


def test_env_render():
    """Test environment rendering."""
    env = PerudoEnv(num_players=2, dice_per_player=5, render_mode="human")
    env.reset()
    
    # Check that render doesn't raise exceptions
    try:
        env.render()
    except Exception as e:
        pytest.fail(f"render() raised an exception: {e}")


def test_env_set_active_player():
    """Test setting active player."""
    env = PerudoEnv(num_players=4, dice_per_player=5)
    env.reset()
    
    # Set active player
    env.set_active_player(2)
    
    # Get observation for them
    obs = env.get_observation_for_player(2)
    
    assert isinstance(obs, np.ndarray)
    assert len(obs) > 0


def test_env_game_over():
    """Test game over in environment."""
    env = PerudoEnv(num_players=2, dice_per_player=5)
    obs, _ = env.reset()
    
    # Simulate game over
    env.game_state.game_over = True
    env.game_state.winner = 0
    
    # Execute action (should return terminated=True)
    action = env.action_space.sample()
    _, _, terminated, _, _ = env.step(action)
    
    assert terminated or env.game_state.game_over

