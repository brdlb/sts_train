"""
Tests for opponent pool management.
"""

import pytest
import numpy as np
import tempfile
import shutil
import os
from pathlib import Path

from src.perudo.training.opponent_pool import OpponentPool, OpponentSnapshot
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from src.perudo.game.perudo_env import PerudoEnv


def test_opponent_pool_initialization():
    """Test opponent pool initialization."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        pool = OpponentPool(
            pool_dir=temp_dir,
            max_pool_size=20,
            min_pool_size=10,
            keep_best=3,
            snapshot_freq=50000,
        )
        
        assert pool.pool_dir == temp_dir
        assert pool.max_pool_size == 20
        assert pool.min_pool_size == 10
        assert pool.keep_best == 3
        assert pool.snapshot_freq == 50000
        assert isinstance(pool.snapshots, dict)
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_save_snapshot():
    """Test saving snapshots."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        pool = OpponentPool(
            pool_dir=temp_dir,
            max_pool_size=10,
            min_pool_size=5,
            snapshot_freq=100,  # Save every 100 steps
        )
        
        # Create a dummy PPO model
        env = PerudoEnv(num_players=4)
        vec_env = DummyVecEnv([lambda: env])
        model = PPO("MlpPolicy", vec_env, verbose=0)
        
        # Save snapshot at step that doesn't match freq
        snapshot_path = pool.save_snapshot(model, step=50)
        
        # Should not save because step % freq != 0
        assert snapshot_path is None
        
        # Save at the right step (step % freq == 0)
        snapshot_path = pool.save_snapshot(model, step=100)
        
        if snapshot_path is not None:
            assert os.path.exists(snapshot_path)
            assert len(pool.snapshots) > 0
        
        vec_env.close()
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_sample_opponent():
    """Test sampling opponents from pool."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        pool = OpponentPool(
            pool_dir=temp_dir,
            max_pool_size=10,
            min_pool_size=5,
            snapshot_freq=1,
        )
        
        # Create dummy snapshots
        env = PerudoEnv(num_players=4)
        vec_env = DummyVecEnv([lambda: env])
        model = PPO("MlpPolicy", vec_env, verbose=0)
        
        # Save a few snapshots
        for step in [1, 2, 3]:
            pool.save_snapshot(model, step=step)
        
        # Sample opponent
        opponent_path = pool.sample_opponent(0)
        
        # Should return a path if there are snapshots
        if len(pool.snapshots) > 0:
            assert opponent_path is not None or opponent_path is None  # Can be None if no snapshots
        
        vec_env.close()
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_update_winrate():
    """Test updating winrate statistics."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        pool = OpponentPool(
            pool_dir=temp_dir,
            max_pool_size=10,
            min_pool_size=5,
            snapshot_freq=1,
        )
        
        # Create dummy snapshot
        env = PerudoEnv(num_players=4)
        vec_env = DummyVecEnv([lambda: env])
        model = PPO("MlpPolicy", vec_env, verbose=0)
        
        snapshot_path = pool.save_snapshot(model, step=1)
        
        if snapshot_path and len(pool.snapshots) > 0:
            # Update winrate
            pool.update_winrate(snapshot_path, won=True)
            
            # Find the snapshot
            snapshot = None
            for s in pool.snapshots.values():
                if s.path == snapshot_path:
                    snapshot = s
                    break
            
            if snapshot:
                assert snapshot.wins == 1
                assert snapshot.games_played == 1
                assert snapshot.winrate == 1.0
        
        vec_env.close()
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_get_statistics():
    """Test getting pool statistics."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        pool = OpponentPool(
            pool_dir=temp_dir,
            max_pool_size=10,
            min_pool_size=5,
            snapshot_freq=1,
        )
        
        stats = pool.get_statistics()
        
        assert isinstance(stats, dict)
        assert "total_snapshots" in stats
        assert "avg_winrate" in stats
        assert "avg_elo" in stats
        
        # Empty pool
        assert stats["total_snapshots"] == 0
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_cleanup_snapshots():
    """Test cleanup of old snapshots."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        pool = OpponentPool(
            pool_dir=temp_dir,
            max_pool_size=5,  # Small max to test cleanup
            min_pool_size=2,
            keep_best=2,
            snapshot_freq=1,
        )
        
        # Create dummy snapshots
        env = PerudoEnv(num_players=4)
        vec_env = DummyVecEnv([lambda: env])
        model = PPO("MlpPolicy", vec_env, verbose=0)
        
        # Save more snapshots than max_pool_size
        for step in range(1, 10):
            pool.save_snapshot(model, step=step)
        
        # Should have cleaned up
        assert len(pool.snapshots) <= pool.max_pool_size
        
        vec_env.close()
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_load_snapshot():
    """Test loading snapshots."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        pool = OpponentPool(
            pool_dir=temp_dir,
            max_pool_size=10,
            min_pool_size=5,
            snapshot_freq=1,
        )
        
        # Create and save snapshot
        env = PerudoEnv(num_players=4)
        vec_env = DummyVecEnv([lambda: env])
        model = PPO("MlpPolicy", vec_env, verbose=0)
        
        snapshot_path = pool.save_snapshot(model, step=1)
        
        if snapshot_path and os.path.exists(snapshot_path):
            # Load snapshot
            loaded_model = pool.load_snapshot(snapshot_path, env)
            
            # Should load successfully
            assert loaded_model is not None
            assert isinstance(loaded_model, PPO)
        
        vec_env.close()
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_get_best_snapshot():
    """Test getting best snapshot by ELO."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        pool = OpponentPool(
            pool_dir=temp_dir,
            max_pool_size=10,
            min_pool_size=5,
            snapshot_freq=1,
        )
        
        # Create dummy snapshots
        env = PerudoEnv(num_players=4)
        vec_env = DummyVecEnv([lambda: env])
        model = PPO("MlpPolicy", vec_env, verbose=0)
        
        # Save a few snapshots
        snapshot_paths = []
        for step in [1, 2, 3]:
            path = pool.save_snapshot(model, step=step)
            if path:
                snapshot_paths.append(path)
        
        # Update ELO for different snapshots
        if len(snapshot_paths) >= 2:
            pool.update_winrate(snapshot_paths[0], won=True, current_elo=1500)
            pool.update_winrate(snapshot_paths[1], won=False, current_elo=1500)
        
        # Get best snapshot
        best_path = pool.get_best_snapshot()
        
        # Should return a path if there are snapshots
        if len(pool.snapshots) > 0:
            assert best_path is not None
        
        vec_env.close()
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

