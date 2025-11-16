"""
Player analysis system for tracking player behavior and tendencies.
"""

from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class FirstBidBluffs:
    """Tracking first bid bluff statistics."""
    count: int = 0
    total: int = 0


@dataclass
class PreRevealTendency:
    """Tracking pre-reveal tendency statistics."""
    bluff_count: int = 0
    strong_hand_count: int = 0
    total: int = 0


@dataclass
class FaceBluffPattern:
    """Tracking bluff patterns for specific dice faces."""
    bluff_count: int = 0
    total_bids: int = 0


@dataclass
class PlayerAnalysis:
    """Player analysis data structure."""
    player_id: int
    first_bid_bluffs: FirstBidBluffs = field(default_factory=FirstBidBluffs)
    pre_reveal_tendency: PreRevealTendency = field(default_factory=PreRevealTendency)
    face_bluff_patterns: Dict[int, FaceBluffPattern] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize face bluff patterns for all dice values (1-6)."""
        if not self.face_bluff_patterns:
            for face in range(1, 7):
                self.face_bluff_patterns[face] = FaceBluffPattern()


def create_initial_player_analysis(player_id: int) -> PlayerAnalysis:
    """
    Create initial player analysis structure.

    Args:
        player_id: Player ID

    Returns:
        Initialized PlayerAnalysis object
    """
    return PlayerAnalysis(player_id=player_id)

