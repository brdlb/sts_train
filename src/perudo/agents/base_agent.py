"""
Base agent class for Perudo game.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional


class BaseAgent(ABC):
    """Base class for agents in Perudo game."""

    def __init__(self, agent_id: int):
        """
        Initialize agent.

        Args:
            agent_id: Unique agent ID
        """
        self.agent_id = agent_id

    @abstractmethod
    def act(self, observation: np.ndarray, deterministic: bool = False) -> int:
        """
        Choose action based on observation.

        Args:
            observation: Observation vector from environment
            deterministic: Whether to use deterministic policy

        Returns:
            Action from action_space
        """
        pass

    @abstractmethod
    def learn(self, *args, **kwargs):
        """Train the agent."""
        pass

    def reset(self):
        """Reset agent state (if there is internal state)."""
        pass
