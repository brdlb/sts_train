"""
Training module for agents.
"""

from .config import Config, GameConfig, TrainingConfig, DEFAULT_CONFIG
from .train import SelfPlayTraining, main
from .opponent_pool import OpponentPool, OpponentSnapshot
from .bot_personality_tracker import BotPersonalityTracker, BotPersonalityStats

__all__ = [
    "Config",
    "GameConfig",
    "TrainingConfig",
    "DEFAULT_CONFIG",
    "SelfPlayTraining",
    "main",
    "OpponentPool",
    "OpponentSnapshot",
    "BotPersonalityTracker",
    "BotPersonalityStats",
]
