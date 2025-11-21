"""
Opponent pool management for self-play training.
"""

import os
import json
import sys
import contextlib
from datetime import datetime
from io import StringIO
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict
from sb3_contrib import MaskablePPO


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
        snapshot_freq: int = 5000,
        elo_k: int = 32,
        opponent_device: str = "cpu",
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
            opponent_device: Device to use for opponent models (default: "cpu" to avoid GPU overhead)
        """
        self.pool_dir = pool_dir
        self.max_pool_size = max_pool_size
        self.min_pool_size = min_pool_size
        self.keep_best = keep_best
        self.snapshot_freq = snapshot_freq
        self.elo_k = elo_k
        self.opponent_device = opponent_device

        # Create directory if it doesn't exist
        os.makedirs(pool_dir, exist_ok=True)

        # Load existing snapshots
        self.snapshots: Dict[str, OpponentSnapshot] = {}
        self._load_metadata()

        # Statistics file
        self.metadata_file = os.path.join(pool_dir, "pool_metadata.json")
        
        # Track current policy ELO rating
        # Default to 1500 if not loaded from metadata
        self.current_policy_elo: float = 1500.0

    def _load_metadata(self):
        """Load metadata about existing snapshots."""
        metadata_file = os.path.join(self.pool_dir, "pool_metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                    # Load current policy ELO if available
                    if "current_policy_elo" in data:
                        loaded_elo = float(data["current_policy_elo"])
                        # Clamp to reasonable bounds
                        self.current_policy_elo = max(0.0, min(3000.0, loaded_elo))
                    for snapshot_id, snapshot_data in data.get("snapshots", {}).items():
                        snapshot = OpponentSnapshot(**snapshot_data)
                        # Fix invalid ELO values (clamp to reasonable bounds)
                        if snapshot.elo < 0 or snapshot.elo > 3000:
                            print(f"Warning: Fixing invalid ELO {snapshot.elo} for snapshot {snapshot_id} to valid range")
                            snapshot.elo = max(0.0, min(3000.0, snapshot.elo))
                        # Check if file still exists
                        if os.path.exists(snapshot.path):
                            self.snapshots[snapshot_id] = snapshot
            except Exception as e:
                print(f"Warning: Could not load pool metadata: {e}")

    def _save_metadata(self):
        """Save metadata about snapshots."""
        data = {
            "current_policy_elo": self.current_policy_elo,
            "snapshots": {
                snapshot_id: asdict(snapshot)
                for snapshot_id, snapshot in self.snapshots.items()
            }
        }
        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2)

    def save_snapshot(
        self, model: MaskablePPO, step: int, prefix: str = "snapshot", force: bool = False
    ) -> Optional[str]:
        """
        Save a snapshot of the current policy.

        Args:
            model: MaskablePPO model to save
            step: Current training step
            prefix: Prefix for snapshot filename
            force: If True, ignore snapshot_freq and always save

        Returns:
            Path to saved snapshot, or None if not saved
        """
        if not force and step % self.snapshot_freq != 0:
            return None

        # Add date to filename
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_id = f"{prefix}_step_{step}_{date_str}"
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

    def sample_opponent(
        self, current_step: int, snapshot_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Sample an opponent from the pool based on win rates.

        Args:
            current_step: Current training step
            snapshot_id: Optional specific snapshot ID to use. If provided,
                        returns that snapshot instead of sampling randomly.

        Returns:
            Path to opponent snapshot, or None if pool is empty or snapshot not found
        """
        if not self.snapshots:
            return None

        # If specific snapshot is requested, return it directly
        if snapshot_id is not None:
            snapshot_path = self.get_snapshot_by_id(snapshot_id)
            if snapshot_path:
                # Update last used
                self.snapshots[snapshot_id].last_used = current_step
                return snapshot_path
            else:
                print(f"Warning: Snapshot ID '{snapshot_id}' not found, falling back to random sampling")
                # Fall through to random sampling

        # Calculate weights based on win rates, ELO, and game count
        # Weighted selection prioritizes:
        # 1. Lower winrate = harder opponent = higher weight
        # 2. More games played = higher confidence = slightly higher weight
        # 3. Higher ELO = stronger opponent = higher weight
        weights = []
        snapshots = []
        for snapshot_id, snapshot in self.snapshots.items():
            # Base weight from winrate: lower winrate = more challenging = higher weight
            # Winrate is probability of current policy winning against this opponent
            # Lower winrate means opponent is stronger, so we want to sample them more
            winrate_weight = 1.0 - snapshot.winrate
            
            # Confidence weight: more games = more reliable winrate
            # Use logarithmic scaling to avoid too much bias toward old snapshots
            confidence_weight = np.log10(max(snapshot.games_played, 1) + 1) / np.log10(100)
            confidence_weight = min(confidence_weight, 1.0)  # Cap at 1.0
            
            # ELO weight: higher ELO = stronger opponent = higher weight
            # Normalize ELO to [0, 1] range (assuming ELO typically in 0-3000 range)
            # Use 1500 as baseline (typical starting ELO)
            elo_weight = (snapshot.elo - 1500.0) / 1500.0
            elo_weight = max(0.0, min(1.0, elo_weight))  # Clamp to [0, 1]
            
            # Combine weights: winrate is primary, confidence and ELO are modifiers
            # Formula: base_weight * (1 + confidence_bonus + elo_bonus)
            combined_weight = winrate_weight * (1.0 + 0.2 * confidence_weight + 0.3 * elo_weight)
            
            # Add small epsilon to avoid zero weights
            weight = max(combined_weight, 0.1)
            weights.append(weight)
            snapshots.append((snapshot_id, snapshot))

        # Normalize weights to probabilities
        total_weight = sum(weights)
        if total_weight == 0 or len(weights) == 0:
            # Fallback to uniform distribution if all weights are zero
            weights = [1.0 / len(snapshots)] * len(snapshots)
        else:
            weights = [w / total_weight for w in weights]

        # Sample based on normalized weights
        selected_idx = np.random.choice(len(snapshots), p=weights, size=1)[0]
        snapshot_id, snapshot = snapshots[selected_idx]

        # Update last used
        snapshot.last_used = current_step

        return snapshot.path

    def update_winrate(
        self, snapshot_path: str, won: bool, current_elo: Optional[float] = None
    ):
        """
        Update win rate statistics for a snapshot and ELO ratings for both players.

        Args:
            snapshot_path: Path to snapshot
            won: Whether current policy won
            current_elo: Current policy ELO rating (uses self.current_policy_elo if None)
        """
        # Find snapshot by path
        snapshot = None
        for s in self.snapshots.values():
            if s.path == snapshot_path:
                snapshot = s
                break

        if snapshot is None:
            return

        # Use stored current policy ELO if not provided
        if current_elo is None:
            current_elo = self.current_policy_elo

        # Update statistics
        snapshot.games_played += 1
        if won:
            snapshot.wins += 1

        # Update win rate
        snapshot.winrate = snapshot.wins / snapshot.games_played

        # Update ELO ratings for both players
        # ELO formula: Expected score = 1 / (1 + 10^((opponent_elo - my_elo) / 400))
        # Calculate expected score for current policy (probability of winning)
        elo_diff = (snapshot.elo - current_elo) / 400.0
        
        # Clamp elo_diff to prevent overflow in 10 ** elo_diff
        # For very large differences, expected_score approaches 0 or 1
        if elo_diff > 10.0:  # Very large difference, snapshot is much stronger
            expected_score_current = 0.0
        elif elo_diff < -10.0:  # Very large difference, current is much stronger
            expected_score_current = 1.0
        else:
            expected_score_current = 1.0 / (1.0 + 10 ** elo_diff)
        
        # Expected score for opponent is the complement
        expected_score_opponent = 1.0 - expected_score_current
        
        # Actual scores
        actual_score_current = 1.0 if won else 0.0
        actual_score_opponent = 1.0 - actual_score_current
        
        # Calculate ELO changes
        elo_change_current = self.elo_k * (actual_score_current - expected_score_current)
        elo_change_opponent = self.elo_k * (actual_score_opponent - expected_score_opponent)
        
        # Update both ELO ratings
        self.current_policy_elo += elo_change_current
        snapshot.elo += elo_change_opponent
        
        # Clamp ELO to reasonable bounds to prevent extreme values
        # Typical ELO ranges: 0-3000 (some systems use negative but it's unusual)
        self.current_policy_elo = max(0.0, min(3000.0, self.current_policy_elo))
        snapshot.elo = max(0.0, min(3000.0, snapshot.elo))

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

    def get_snapshot_by_id(self, snapshot_id: str) -> Optional[str]:
        """
        Get a snapshot path by its ID.

        Args:
            snapshot_id: ID of the snapshot (e.g., "snapshot_step_200000")

        Returns:
            Path to snapshot, or None if not found
        """
        if snapshot_id in self.snapshots:
            return self.snapshots[snapshot_id].path
        return None

    def get_snapshot_by_step(self, step: int) -> Optional[str]:
        """
        Get a snapshot path by training step.

        Args:
            step: Training step number

        Returns:
            Path to snapshot, or None if not found
        """
        for snapshot in self.snapshots.values():
            if snapshot.step == step:
                return snapshot.path
        return None

    def list_snapshots(self) -> Dict[str, Dict]:
        """
        List all available snapshots with their metadata.

        Returns:
            Dictionary mapping snapshot_id to snapshot metadata
        """
        return {
            snapshot_id: {
                "path": snapshot.path,
                "step": snapshot.step,
                "winrate": snapshot.winrate,
                "games_played": snapshot.games_played,
                "wins": snapshot.wins,
                "elo": snapshot.elo,
                "last_used": snapshot.last_used,
            }
            for snapshot_id, snapshot in self.snapshots.items()
        }

    def load_snapshot(self, snapshot_path: str, env) -> Optional[MaskablePPO]:
        """
        Load a snapshot model.

        Args:
            snapshot_path: Path to snapshot
            env: Environment for loading model

        Returns:
            Loaded MaskablePPO model, or None if loading failed
        """
        if not os.path.exists(snapshot_path):
            return None

        # Suppress SB3 wrapping messages
        with contextlib.redirect_stdout(StringIO()):
            # Load opponent model on CPU to avoid GPU overhead from single-observation predictions
            model = MaskablePPO.load(snapshot_path, env=env, device=self.opponent_device)
        return model

    @staticmethod
    def load_model_from_path(model_path: str, env, device: str = "cpu") -> Optional[MaskablePPO]:
        """
        Load a model from any path (not necessarily in the pool).

        This allows loading models that are not registered in the pool metadata,
        for example models saved outside the opponent pool system.

        Args:
            model_path: Path to model file (.zip)
            env: Environment for loading model
            device: Device to load model on (default: "cpu" to avoid GPU overhead)

        Returns:
            Loaded MaskablePPO model, or None if loading failed
        """
        if not os.path.exists(model_path):
            print(f"Error: Model path does not exist: {model_path}")
            return None

        # Suppress SB3 wrapping messages
        with contextlib.redirect_stdout(StringIO()):
            # Load opponent model on CPU to avoid GPU overhead from single-observation predictions
            model = MaskablePPO.load(model_path, env=env, device=device)
        return model

    def get_current_policy_elo(self) -> float:
        """
        Get current policy ELO rating.

        Returns:
            Current policy ELO rating
        """
        return self.current_policy_elo

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
                "current_policy_elo": self.current_policy_elo,
            }

        winrates = [s.winrate for s in self.snapshots.values()]
        elos = [s.elo for s in self.snapshots.values()]

        return {
            "total_snapshots": len(self.snapshots),
            "avg_winrate": np.mean(winrates),
            "avg_elo": np.mean(elos),
            "best_elo": max(elos),
            "min_elo": min(elos),
            "current_policy_elo": self.current_policy_elo,
        }

