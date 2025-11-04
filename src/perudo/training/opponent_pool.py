"""
Opponent pool management for self-play training.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict
from stable_baselines3 import PPO


@dataclass
class OpponentSnapshot:
    """Snapshot of an opponent policy."""

    path: str
    step: int
    winrate: float = 0.5  # Win rate against current policy
    games_played: int = 0
    wins: int = 0
    last_used: int = 0  # Step when last used
    elo: float = 1500.0  # ELO rating


class OpponentPool:
    """
    Manages a pool of opponent policy snapshots for self-play.

    Features:
    - Save snapshots periodically
    - Track win rates against each snapshot
    - Weighted sampling based on win rates
    - Automatic cleanup of old snapshots
    - Keep best snapshots
    """

    def __init__(
        self,
        pool_dir: str,
        max_pool_size: int = 20,
        min_pool_size: int = 10,
        keep_best: int = 3,
        snapshot_freq: int = 50000,
        elo_k: int = 32,
    ):
        """
        Initialize opponent pool.

        Args:
            pool_dir: Directory to store snapshots
            max_pool_size: Maximum number of snapshots to keep
            min_pool_size: Minimum number of snapshots to keep
            keep_best: Number of best snapshots to always keep
            snapshot_freq: Frequency of saving snapshots (in steps)
            elo_k: ELO K-factor for rating updates
        """
        self.pool_dir = pool_dir
        self.max_pool_size = max_pool_size
        self.min_pool_size = min_pool_size
        self.keep_best = keep_best
        self.snapshot_freq = snapshot_freq
        self.elo_k = elo_k

        # Create directory if it doesn't exist
        os.makedirs(pool_dir, exist_ok=True)

        # Load existing snapshots
        self.snapshots: Dict[str, OpponentSnapshot] = {}
        self._load_metadata()

        # Statistics file
        self.metadata_file = os.path.join(pool_dir, "pool_metadata.json")

    def _load_metadata(self):
        """Load metadata about existing snapshots."""
        metadata_file = os.path.join(self.pool_dir, "pool_metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                    for snapshot_id, snapshot_data in data.get("snapshots", {}).items():
                        snapshot = OpponentSnapshot(**snapshot_data)
                        # Check if file still exists
                        if os.path.exists(snapshot.path):
                            self.snapshots[snapshot_id] = snapshot
            except Exception as e:
                print(f"Warning: Could not load pool metadata: {e}")

    def _save_metadata(self):
        """Save metadata about snapshots."""
        data = {
            "snapshots": {
                snapshot_id: asdict(snapshot)
                for snapshot_id, snapshot in self.snapshots.items()
            }
        }
        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2)

    def save_snapshot(
        self, model: PPO, step: int, prefix: str = "snapshot"
    ) -> Optional[str]:
        """
        Save a snapshot of the current policy.

        Args:
            model: PPO model to save
            step: Current training step
            prefix: Prefix for snapshot filename

        Returns:
            Path to saved snapshot, or None if not saved
        """
        if step % self.snapshot_freq != 0:
            return None

        snapshot_id = f"{prefix}_step_{step}"
        snapshot_path = os.path.join(self.pool_dir, f"{snapshot_id}.zip")

        # Save model
        model.save(snapshot_path)

        # Create snapshot metadata
        snapshot = OpponentSnapshot(
            path=snapshot_path,
            step=step,
            winrate=0.5,
            games_played=0,
            wins=0,
            last_used=step,
            elo=1500.0,
        )

        self.snapshots[snapshot_id] = snapshot

        # Clean up old snapshots
        self._cleanup_snapshots(step)

        # Save metadata
        self._save_metadata()

        print(f"Saved snapshot: {snapshot_path} (step {step})")
        return snapshot_path

    def sample_opponent(self, current_step: int) -> Optional[str]:
        """
        Sample an opponent from the pool based on win rates.

        Args:
            current_step: Current training step

        Returns:
            Path to opponent snapshot, or None if pool is empty
        """
        if not self.snapshots:
            return None

        # Calculate weights based on win rates
        # Lower win rate = higher weight (harder opponents)
        weights = []
        snapshots = []
        for snapshot_id, snapshot in self.snapshots.items():
            # Use winrate to weight: lower winrate = more challenging = higher weight
            # But we want to sample harder opponents more often
            # So weight = 1 - winrate (inverted)
            weight = 1.0 - snapshot.winrate
            # Add small epsilon to avoid zero weights
            weight = max(weight, 0.1)
            weights.append(weight)
            snapshots.append((snapshot_id, snapshot))

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0 / len(weights)] * len(weights)
        else:
            weights = [w / total_weight for w in weights]

        # Sample based on weights
        selected_idx = np.random.choice(len(snapshots), p=weights, size=1)[0]
        snapshot_id, snapshot = snapshots[selected_idx]

        # Update last used
        snapshot.last_used = current_step

        return snapshot.path

    def update_winrate(
        self, snapshot_path: str, won: bool, current_elo: float = 1500.0
    ):
        """
        Update win rate statistics for a snapshot.

        Args:
            snapshot_path: Path to snapshot
            won: Whether current policy won
            current_elo: Current policy ELO rating
        """
        # Find snapshot by path
        snapshot = None
        for s in self.snapshots.values():
            if s.path == snapshot_path:
                snapshot = s
                break

        if snapshot is None:
            return

        # Update statistics
        snapshot.games_played += 1
        if won:
            snapshot.wins += 1

        # Update win rate
        snapshot.winrate = snapshot.wins / snapshot.games_played

        # Update ELO
        expected_score = 1.0 / (1.0 + 10 ** ((snapshot.elo - current_elo) / 400.0))
        actual_score = 1.0 if won else 0.0
        elo_change = self.elo_k * (actual_score - expected_score)
        snapshot.elo += elo_change

        # Save metadata
        self._save_metadata()

    def _cleanup_snapshots(self, current_step: int):
        """Clean up old snapshots, keeping best ones."""
        if len(self.snapshots) <= self.max_pool_size:
            return

        # Sort snapshots by various criteria
        snapshot_list = list(self.snapshots.items())

        # Sort by ELO (best first)
        snapshot_list.sort(key=lambda x: x[1].elo, reverse=True)

        # Keep best N snapshots
        best_snapshots = set(
            snapshot_id for snapshot_id, _ in snapshot_list[: self.keep_best]
        )

        # For remaining snapshots, sort by last_used (oldest first)
        # and keep those that are more recent
        remaining_snapshots = [
            (snapshot_id, snapshot)
            for snapshot_id, snapshot in snapshot_list[self.keep_best :]
        ]
        remaining_snapshots.sort(key=lambda x: x[1].last_used, reverse=True)

        # Keep up to max_pool_size - keep_best
        keep_count = self.max_pool_size - self.keep_best
        keep_snapshots = set(
            snapshot_id for snapshot_id, _ in remaining_snapshots[:keep_count]
        )

        # Remove snapshots that are not in keep sets
        snapshots_to_remove = []
        for snapshot_id, snapshot in self.snapshots.items():
            if snapshot_id not in best_snapshots and snapshot_id not in keep_snapshots:
                snapshots_to_remove.append(snapshot_id)

        # Remove snapshots
        for snapshot_id in snapshots_to_remove:
            snapshot = self.snapshots[snapshot_id]
            # Optionally delete file
            # os.remove(snapshot.path)
            del self.snapshots[snapshot_id]

        print(f"Cleaned up {len(snapshots_to_remove)} snapshots")

    def get_best_snapshot(self) -> Optional[str]:
        """
        Get the best snapshot by ELO.

        Returns:
            Path to best snapshot, or None if pool is empty
        """
        if not self.snapshots:
            return None

        best_snapshot = max(self.snapshots.values(), key=lambda s: s.elo)
        return best_snapshot.path

    def load_snapshot(self, snapshot_path: str, env) -> Optional[PPO]:
        """
        Load a snapshot model.

        Args:
            snapshot_path: Path to snapshot
            env: Environment for loading model

        Returns:
            Loaded PPO model, or None if loading failed
        """
        if not os.path.exists(snapshot_path):
            return None

        try:
            model = PPO.load(snapshot_path, env=env)
            return model
        except Exception as e:
            print(f"Error loading snapshot {snapshot_path}: {e}")
            return None

    def get_statistics(self) -> Dict:
        """
        Get statistics about the pool.

        Returns:
            Dictionary with pool statistics
        """
        if not self.snapshots:
            return {
                "total_snapshots": 0,
                "avg_winrate": 0.0,
                "avg_elo": 0.0,
                "best_elo": 0.0,
            }

        winrates = [s.winrate for s in self.snapshots.values()]
        elos = [s.elo for s in self.snapshots.values()]

        return {
            "total_snapshots": len(self.snapshots),
            "avg_winrate": np.mean(winrates),
            "avg_elo": np.mean(elos),
            "best_elo": max(elos),
            "min_elo": min(elos),
        }

