"""
Training module for agents.
"""

from .config import Config, GameConfig, TrainingConfig, DEFAULT_CONFIG
from .train import SelfPlayTraining, main
from .opponent_pool import OpponentPool, OpponentSnapshot

__all__ = [
    "Config",
    "GameConfig",
    "TrainingConfig",
    "DEFAULT_CONFIG",
    "SelfPlayTraining",
    "main",
    "OpponentPool",
    "OpponentSnapshot",
]
