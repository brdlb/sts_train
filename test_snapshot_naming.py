
import os
import shutil
import sys
from unittest.mock import MagicMock
from datetime import datetime

# Add src to path
sys.path.append(os.getcwd())

from src.perudo.training.opponent_pool import OpponentPool

def test_snapshot_naming():
    # Setup
    pool_dir = "test_pool_dir"
    if os.path.exists(pool_dir):
        shutil.rmtree(pool_dir)
    os.makedirs(pool_dir)

    pool = OpponentPool(pool_dir=pool_dir, snapshot_freq=1)
    
    # Mock model
    model = MagicMock()
    model.save = MagicMock()
    
    # Test save_snapshot
    step = 100
    snapshot_path = pool.save_snapshot(model, step, force=True)
    
    print(f"Snapshot path: {snapshot_path}")
    
    # Verify filename format
    filename = os.path.basename(snapshot_path)
    expected_prefix = f"snapshot_step_{step}_"
    
    if filename.startswith(expected_prefix) and filename.endswith(".zip"):
        # Check if date part is present and valid
        date_part = filename[len(expected_prefix):-4]
        try:
            datetime.strptime(date_part, "%Y%m%d_%H%M%S")
            print("SUCCESS: Filename contains valid date format.")
        except ValueError:
            print(f"FAILURE: Filename date part '{date_part}' is invalid.")
    else:
        print(f"FAILURE: Filename '{filename}' does not match expected format '{expected_prefix}<DATE>.zip'")

    # Cleanup
    if os.path.exists(pool_dir):
        shutil.rmtree(pool_dir)

if __name__ == "__main__":
    test_snapshot_naming()
