"""
Training module for agents.
"""

from .config import Config, GameConfig, TrainingConfig, DEFAULT_CONFIG
from .train import SelfPlayTraining
from .cli import main
from .opponent_pool import OpponentPool, OpponentSnapshot
from .callbacks import (
    AdaptiveEntropyCallback,
    SelfPlayTrainingCallback,
    ModelUpdateProgressCallback,
    WinnerTrajectoryCollectorCallback,
)
from .utils import (
    get_device,
    linear_schedule,
    find_latest_model,
    restore_model_from_opponent_pool,
)

__all__ = [
    "Config",
    "GameConfig",
    "TrainingConfig",
    "DEFAULT_CONFIG",
    "SelfPlayTraining",
    "main",
    "OpponentPool",
    "OpponentSnapshot",
    "AdaptiveEntropyCallback",
    "SelfPlayTrainingCallback",
    "ModelUpdateProgressCallback",
    "WinnerTrajectoryCollectorCallback",
    "get_device",
    "linear_schedule",
    "find_latest_model",
    "restore_model_from_opponent_pool",
]
