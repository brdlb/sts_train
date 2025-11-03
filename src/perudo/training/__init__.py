"""
Training module for agents.
"""

from .config import Config, GameConfig, TrainingConfig, DEFAULT_CONFIG
from .train import SelfPlayTraining, train_single_agent_loop, main

__all__ = [
    "Config",
    "GameConfig",
    "TrainingConfig",
    "DEFAULT_CONFIG",
    "SelfPlayTraining",
    "train_single_agent_loop",
    "main",
]
