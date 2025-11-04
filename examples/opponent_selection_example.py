"""
Example of how to use explicit opponent model selection.

This script demonstrates different ways to specify which opponent models to use.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.perudo.training.opponent_pool import OpponentPool
from src.perudo.game.perudo_vec_env import PerudoMultiAgentVecEnv


def example_list_snapshots():
    """Example: List all available snapshots."""
    pool_dir = "models/opponent_pool"
    opponent_pool = OpponentPool(pool_dir=pool_dir)
    
    # List all snapshots
    snapshots = opponent_pool.list_snapshots()
    print("Available snapshots:")
    for snapshot_id, metadata in snapshots.items():
        print(f"  {snapshot_id}:")
        print(f"    Step: {metadata['step']}")
        print(f"    Winrate: {metadata['winrate']:.2%}")
        print(f"    ELO: {metadata['elo']:.1f}")
        print(f"    Path: {metadata['path']}")
        print()


def example_get_specific_snapshot():
    """Example: Get a specific snapshot by ID."""
    pool_dir = "models/opponent_pool"
    opponent_pool = OpponentPool(pool_dir=pool_dir)
    
    # Get snapshot by ID
    snapshot_id = "snapshot_step_200000"
    snapshot_path = opponent_pool.get_snapshot_by_id(snapshot_id)
    
    if snapshot_path:
        print(f"Found snapshot: {snapshot_path}")
    else:
        print(f"Snapshot {snapshot_id} not found")
    
    # Get snapshot by step number
    step = 200000
    snapshot_path = opponent_pool.get_snapshot_by_step(step)
    
    if snapshot_path:
        print(f"Found snapshot at step {step}: {snapshot_path}")
    else:
        print(f"No snapshot found at step {step}")


def example_specify_opponent_in_reset():
    """Example: Specify opponents when resetting environment."""
    pool_dir = "models/opponent_pool"
    opponent_pool = OpponentPool(pool_dir=pool_dir)
    
    # Create vectorized environment
    vec_env = PerudoMultiAgentVecEnv(
        num_envs=2,
        num_players=4,
        dice_per_player=5,
        opponent_pool=opponent_pool,
    )
    
    # Option 1: Use random sampling (default behavior)
    obs = vec_env.reset()
    print("Reset with random opponents")
    
    # Option 2: Specify specific snapshot IDs for opponents
    # For 4 players, we need 3 opponents (agents 1, 2, 3)
    # Use specific snapshot for first opponent, random for others
    options = [
        {
            "opponent_snapshot_ids": [
                "snapshot_step_200000",  # Opponent 1 (agent 1)
                None,  # Opponent 2 (agent 2) - random
                "snapshot_step_300000",  # Opponent 3 (agent 3)
            ]
        },
        None,  # Second environment uses random opponents
    ]
    obs = vec_env.reset(options=options)
    print("Reset with specified opponents")
    
    # Option 3: All opponents use the same snapshot
    options = [
        {
            "opponent_snapshot_ids": [
                "snapshot_step_200000",
                "snapshot_step_200000",
                "snapshot_step_200000",
            ]
        },
    ]
    obs = vec_env.reset(options=options)
    print("Reset with all opponents using same snapshot")


def example_load_model_from_anywhere():
    """Example: Load a model from any path (not in pool)."""
    from src.perudo.game.perudo_env import PerudoEnv
    
    # Create a simple environment
    env = PerudoEnv(num_players=4, dice_per_player=5)
    
    # Load model from any path (e.g., from main model directory)
    model_path = "models/perudo_model_800000_steps.zip"
    model = OpponentPool.load_model_from_path(model_path, env)
    
    if model:
        print(f"Successfully loaded model from {model_path}")
    else:
        print(f"Failed to load model from {model_path}")


def example_use_best_snapshot():
    """Example: Always use the best snapshot by ELO."""
    pool_dir = "models/opponent_pool"
    opponent_pool = OpponentPool(pool_dir=pool_dir)
    
    # Get best snapshot
    best_snapshot_path = opponent_pool.get_best_snapshot()
    
    if best_snapshot_path:
        print(f"Best snapshot: {best_snapshot_path}")
        
        # Get its ID to use in reset
        snapshots = opponent_pool.list_snapshots()
        for snapshot_id, metadata in snapshots.items():
            if metadata["path"] == best_snapshot_path:
                print(f"Best snapshot ID: {snapshot_id}")
                break
    else:
        print("No snapshots available")


if __name__ == "__main__":
    print("=== Example 1: List all snapshots ===")
    example_list_snapshots()
    
    print("\n=== Example 2: Get specific snapshot ===")
    example_get_specific_snapshot()
    
    print("\n=== Example 3: Use best snapshot ===")
    example_use_best_snapshot()
    
    print("\n=== Example 4: Load model from anywhere ===")
    example_load_model_from_anywhere()
    
    print("\n=== Example 5: Specify opponents in reset ===")
    print("(This requires a running environment)")
    # example_specify_opponent_in_reset()

