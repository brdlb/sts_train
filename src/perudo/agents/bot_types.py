"""
Type definitions for rule-based bots.
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple
from enum import Enum


class GameStage(Enum):
    """Game stage based on total dice and active players."""
    CHAOS = "CHAOS"
    POSITIVE = "POSITIVE"
    TENSE = "TENSE"
    KNIFE_FIGHT = "KNIFE_FIGHT"
    DUEL = "DUEL"


@dataclass
class BotAffinities:
    """
    Bot affinities for different analysis types.
    
    These numbers act as multipliers for analysis results.
    1.0 is standard. >1.0 is specialty. <1.0 is weakness.
    """
    first_bid_analysis: float = 1.0
    pre_reveal_analysis: float = 1.0
    face_pattern_analysis: float = 1.0


@dataclass
class BotPersonality:
    """Bot personality configuration."""
    name: str
    description: str
    avatar: str
    skill_level: Literal["EASY", "MEDIUM", "HARD"]
    affinities: BotAffinities


@dataclass
class FirstBidBluffs:
    """Statistics about first bid bluffs."""
    count: int = 0
    total: int = 0


@dataclass
class PreRevealTendency:
    """Statistics about pre-reveal tendencies."""
    bluff_count: int = 0
    strong_hand_count: int = 0
    total: int = 0


@dataclass
class FaceBluffPattern:
    """Pattern of bluffing on a specific face."""
    bluff_count: int = 0
    total_bids: int = 0


@dataclass
class PlayerAnalysis:
    """
    Analysis data for a player.
    
    Tracks:
    - First bid bluffs: whether player bluffs on their opening bid
    - Pre-reveal tendency: whether player tends to bluff or have strong hands before reveal
    - Face bluff patterns: patterns of bluffing on specific dice faces
    """
    player_id: str
    
    # Metric 1: First bid bluff analysis
    first_bid_bluffs: FirstBidBluffs = field(default_factory=FirstBidBluffs)
    
    # Metric 2: Pre-reveal tendency analysis
    pre_reveal_tendency: PreRevealTendency = field(default_factory=PreRevealTendency)
    
    # Metric 3: Face bluff patterns (key: dice value as string "1"-"6")
    face_bluff_patterns: Dict[str, FaceBluffPattern] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize face bluff patterns for all dice values."""
        if not self.face_bluff_patterns:
            for face in range(1, 7):
                self.face_bluff_patterns[str(face)] = FaceBluffPattern()


@dataclass
class Bid:
    """A bid in the game."""
    quantity: int
    face: int  # Dice value (1-6, where 1 is joker/pasari)


@dataclass
class BotDecision:
    """
    Decision made by a bot.
    
    decision: 'BID' means place a bid, 'DUDO' means challenge, 'CALZA' means believe
    bid: The bid to place (only if decision is 'BID')
    thought: Internal reasoning (for debugging/logging)
    dialogue: What the bot would say (for UI)
    """
    decision: Literal["BID", "DUDO", "CALZA"]
    bid: Optional[Bid] = None
    thought: str = ""
    dialogue: str = ""







