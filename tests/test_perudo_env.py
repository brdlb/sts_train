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
    """Test observation shape with agent_id."""
    env = PerudoEnv(num_players=2, dice_per_player=5, history_length=10)
    obs, _ = env.reset()
    
    # New format: agent_id(num_players) + current_bid(2) + history(history_length*3) +
    # dice_count(num_players) + current_player(1) + palifico(num_players) + believe(1) + player_dice(5)
    expected_size = 2 + 2 + 10 * 3 + 2 + 1 + 2 + 1 + 5
    assert obs.shape == (expected_size,)


def test_env_action_space():
    """Test actions space size."""
    env = PerudoEnv(num_players=2, dice_per_player=5, max_quantity=30)
    
    # Size should be: 2 (challenge, believe) + 30 * 6 (bids)
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


def test_agent_id_in_observation():
    """Test that agent_id is included in observation."""
    env = PerudoEnv(num_players=4, dice_per_player=5)
    
    # Get observations for different players
    obs_0, _ = env.reset()
    env.set_active_player(0)
    obs_0 = env.get_observation_for_player(0)
    
    env.set_active_player(1)
    obs_1 = env.get_observation_for_player(1)
    
    env.set_active_player(2)
    obs_2 = env.get_observation_for_player(2)
    
    env.set_active_player(3)
    obs_3 = env.get_observation_for_player(3)
    
    # Check that agent_id one-hot encoding is correct
    # First 4 values should be agent_id one-hot
    assert obs_0[0] == 1.0  # Agent 0
    assert obs_0[1] == 0.0
    assert obs_0[2] == 0.0
    assert obs_0[3] == 0.0
    
    assert obs_1[0] == 0.0
    assert obs_1[1] == 1.0  # Agent 1
    assert obs_1[2] == 0.0
    assert obs_1[3] == 0.0
    
    assert obs_2[0] == 0.0
    assert obs_2[1] == 0.0
    assert obs_2[2] == 1.0  # Agent 2
    assert obs_2[3] == 0.0
    
    assert obs_3[0] == 0.0
    assert obs_3[1] == 0.0
    assert obs_3[2] == 0.0
    assert obs_3[3] == 1.0  # Agent 3


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


def test_next_round_starts_with_player_who_lost_die():
    """Test that next round starts with player who lost die in challenge."""
    env = PerudoEnv(num_players=3, dice_per_player=5)
    obs, _ = env.reset()
    
    # Set up a bid
    env.game_state.set_bid(0, 10, 3)  # Player 0 makes bid
    env.game_state.current_player = 1
    env.set_active_player(1)
    
    # Set dice so challenge will succeed (bid too high)
    env.game_state.player_dice = [[2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]]  # No 3s
    
    # Challenge action (action 0 = challenge)
    challenge_action = 0
    obs, reward, terminated, truncated, info = env.step(challenge_action)
    
    # Player 0 (bid maker) should lose die and start next round
    assert env.game_state.current_player == 0
    assert env.game_state.player_dice_count[0] == 4  # Lost one die


def test_next_round_starts_with_player_who_gained_die():
    """Test that next round starts with player who gained die in believe."""
    env = PerudoEnv(num_players=2, dice_per_player=5)
    env.game_state.player_dice_count[0] = 3  # Bid maker has 3 dice
    env.game_state.player_dice_count[1] = 4  # Believer has 4 dice
    obs, _ = env.reset()
    
    # Set up a bid
    env.game_state.set_bid(0, 5, 3)  # Player 0 makes bid
    env.game_state.current_player = 1
    env.set_active_player(1)
    
    # Set dice so count exactly equals bid (5 threes)
    env.game_state.player_dice = [[2, 2, 3], [1, 1, 3, 3]]
    
    # Find believe action
    # Action 0 = challenge, action 1 = believe
    believe_action = 1
    obs, reward, terminated, truncated, info = env.step(believe_action)
    
    # Player 1 (believer) should gain die and start next round
    assert env.game_state.current_player == 1
    assert env.game_state.player_dice_count[1] == 5  # Gained one die


def test_next_round_starts_with_player_who_lost_die_in_believe():
    """Test that next round starts with player who lost die when believe failed."""
    env = PerudoEnv(num_players=2, dice_per_player=5)
    obs, _ = env.reset()
    
    # Set up a bid
    env.game_state.set_bid(0, 5, 3)  # Player 0 makes bid
    env.game_state.current_player = 1
    env.set_active_player(1)
    
    # Set dice so count doesn't equal bid (more than bid)
    env.game_state.player_dice = [[1, 3, 3, 3, 3], [1, 3, 3, 3, 5]]  # 9 threes total
    
    # Find believe action
    believe_action = 1
    obs, reward, terminated, truncated, info = env.step(believe_action)
    
    # Player 1 (believer) should lose die and start next round
    assert env.game_state.current_player == 1
    assert env.game_state.player_dice_count[1] == 4  # Lost one die

