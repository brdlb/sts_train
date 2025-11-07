"""
Tests for vectorized Perudo environment with multiple tables.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from src.perudo.game.perudo_vec_env import PerudoMultiAgentVecEnv
from src.perudo.training.opponent_pool import OpponentPool


def test_vec_env_initialization():
    """Test vectorized environment initialization."""
    vec_env = PerudoMultiAgentVecEnv(
        num_envs=4,
        num_players=4,
        dice_per_player=5,
    )
    
    assert vec_env.num_envs == 4
    assert vec_env.num_players == 4
    assert len(vec_env.envs) == 4
    assert vec_env.observation_space is not None
    assert vec_env.action_space is not None


def test_vec_env_reset():
    """Test vectorized environment reset."""
    vec_env = PerudoMultiAgentVecEnv(
        num_envs=2,
        num_players=4,
        dice_per_player=5,
    )
    
    obs = vec_env.reset()
    
    assert isinstance(obs, dict)
    assert "bid_history" in obs
    assert "static_info" in obs
    assert obs['bid_history'].shape[0] == 2
    assert obs['static_info'].shape[0] == 2
    assert obs['bid_history'].dtype == np.int32
    assert obs['static_info'].dtype == np.float32


def test_vec_env_step():
    """Test vectorized environment step."""
    vec_env = PerudoMultiAgentVecEnv(
        num_envs=2,
        num_players=4,
        dice_per_player=5,
    )
    
    obs = vec_env.reset()
    
    # Sample actions for both environments
    actions = np.array([vec_env.action_space.sample() for _ in range(2)])
    
    # Step environments
    next_obs, rewards, dones, infos = vec_env.step(actions)
    
    assert isinstance(next_obs, dict)
    assert "bid_history" in next_obs
    assert "static_info" in next_obs
    assert next_obs['bid_history'].shape[0] == 2
    assert next_obs['static_info'].shape[0] == 2
    assert isinstance(rewards, np.ndarray)
    assert rewards.shape == (2,)
    assert isinstance(dones, np.ndarray)
    assert dones.shape == (2,)
    assert isinstance(infos, list)
    assert len(infos) == 2


def test_vec_env_with_opponent_pool():
    """Test vectorized environment with opponent pool."""
    # Create temporary directory for opponent pool
    temp_dir = tempfile.mkdtemp()
    
    try:
        opponent_pool = OpponentPool(
            pool_dir=temp_dir,
            max_pool_size=10,
            min_pool_size=5,
            keep_best=2,
        )
        
        vec_env = PerudoMultiAgentVecEnv(
            num_envs=2,
            num_players=4,
            dice_per_player=5,
            opponent_pool=opponent_pool,
        )
        
        # Reset should sample opponents
        obs = vec_env.reset()
        
        assert isinstance(obs, dict)
        assert "bid_history" in obs
        assert "static_info" in obs
        assert obs['bid_history'].shape[0] == 2
        assert obs['static_info'].shape[0] == 2
        
        # Check that opponent models are set
        assert len(vec_env.opponent_models) == 2
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_vec_env_opponent_turns():
    """Test that opponents take turns correctly."""
    vec_env = PerudoMultiAgentVecEnv(
        num_envs=1,
        num_players=4,
        dice_per_player=5,
    )
    
    obs = vec_env.reset()
    
    # Track how many steps we've taken
    steps = 0
    max_steps = 100  # Prevent infinite loop
    
    while steps < max_steps:
        # Check if it's learning agent's turn (agent 0)
        if vec_env.active_agent_ids[0] == 0:
            # Learning agent's turn - take action
            action = np.array([vec_env.action_space.sample()])
            obs, rewards, dones, infos = vec_env.step(action)
            steps += 1
            
            if dones[0]:
                break
        else:
            # Opponent's turn - should be handled automatically
            # Just wait a bit to see if it progresses
            steps += 1
            if steps >= max_steps:
                break
    
    # Should have made some progress
    assert steps > 0


def test_vec_env_close():
    """Test closing vectorized environment."""
    vec_env = PerudoMultiAgentVecEnv(
        num_envs=2,
        num_players=4,
        dice_per_player=5,
    )
    
    # Should not raise exception
    vec_env.close()


def test_vec_env_get_attr():
    """Test getting attributes from environments."""
    vec_env = PerudoMultiAgentVecEnv(
        num_envs=2,
        num_players=4,
        dice_per_player=5,
    )
    
    # Get num_players attribute
    num_players = vec_env.get_attr("num_players")
    assert len(num_players) == 2
    assert all(n >= 3 and n <= 8 for n in num_players)


def test_vec_env_set_attr():
    """Test setting attributes in environments."""
    vec_env = PerudoMultiAgentVecEnv(
        num_envs=2,
        num_players=4,
        dice_per_player=5,
    )
    
    # Set render_mode
    vec_env.set_attr("render_mode", "human")
    
    # Verify it was set
    render_modes = vec_env.get_attr("render_mode")
    assert all(m == "human" for m in render_modes)
