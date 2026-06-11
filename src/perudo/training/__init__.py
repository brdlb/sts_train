"""
Training module for agents.
"""

from .config import Config, GameConfig, TrainingConfig, DEFAULT_CONFIG

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


def __getattr__(name):
    if name == "SelfPlayTraining":
        from .train import SelfPlayTraining

        return SelfPlayTraining
    if name == "main":
        from .cli import main

        return main
    if name in {"OpponentPool", "OpponentSnapshot"}:
        from .opponent_pool import OpponentPool, OpponentSnapshot

        return {"OpponentPool": OpponentPool, "OpponentSnapshot": OpponentSnapshot}[name]
    if name in {
        "AdaptiveEntropyCallback",
        "SelfPlayTrainingCallback",
        "ModelUpdateProgressCallback",
        "WinnerTrajectoryCollectorCallback",
    }:
        from .callbacks import (
            AdaptiveEntropyCallback,
            SelfPlayTrainingCallback,
            ModelUpdateProgressCallback,
            WinnerTrajectoryCollectorCallback,
        )

        return {
            "AdaptiveEntropyCallback": AdaptiveEntropyCallback,
            "SelfPlayTrainingCallback": SelfPlayTrainingCallback,
            "ModelUpdateProgressCallback": ModelUpdateProgressCallback,
            "WinnerTrajectoryCollectorCallback": WinnerTrajectoryCollectorCallback,
        }[name]
    if name in {
        "get_device",
        "linear_schedule",
        "find_latest_model",
        "restore_model_from_opponent_pool",
    }:
        from .utils import (
            find_latest_model,
            get_device,
            linear_schedule,
            restore_model_from_opponent_pool,
        )

        return {
            "get_device": get_device,
            "linear_schedule": linear_schedule,
            "find_latest_model": find_latest_model,
            "restore_model_from_opponent_pool": restore_model_from_opponent_pool,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
