"""
RL agent based on Stable Baselines3.
"""

import numpy as np
from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .base_agent import BaseAgent
from ..game.perudo_env import PerudoEnv


class RLAgent(BaseAgent):
    """RL agent using Stable Baselines3."""

    def __init__(
        self,
        agent_id: int,
        env: PerudoEnv,
        model: Optional[PPO] = None,
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        verbose: int = 1,
    ):
        """
        Initialize RL agent.

        Args:
            agent_id: Unique agent ID
            env: Perudo environment
            model: Existing PPO model (if any)
            policy: Policy type
            learning_rate: Learning rate
            n_steps: Number of steps to collect data before update
            batch_size: Batch size for training
            n_epochs: Number of epochs to update on one data collection
            gamma: Discount factor
            gae_lambda: GAE parameter (Generalized Advantage Estimation)
            clip_range: Clipping parameter for PPO
            ent_coef: Entropy coefficient (for exploration)
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm
            verbose: Verbosity level
        """
        super().__init__(agent_id)

        self.env = env
        self.verbose = verbose

        if model is not None:
            self.model = model
            self.vec_env = None  # If model already created, vec_env not needed
        else:
            # Create environment wrapper for SB3
            def make_env():
                return env

            self.vec_env = DummyVecEnv([make_env])

            # Create PPO model
            self.model = PPO(
                policy=policy,
                env=self.vec_env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                verbose=verbose,
            )

    def act(self, observation: np.ndarray, deterministic: bool = False) -> int:
        """
        Choose action based on observation.

        Args:
            observation: Observation vector from environment
            deterministic: Whether to use deterministic policy

        Returns:
            Action from action_space
        """
        # Convert observation to format for SB3
        # SB3 expects array of observations, so add dimension
        obs_array = observation.reshape(1, -1)

        action, _ = self.model.predict(obs_array, deterministic=deterministic)
        return int(action[0])

    def learn(self, total_timesteps: int, tb_log_name: Optional[str] = None, **kwargs):
        """
        Train the agent.

        Args:
            total_timesteps: Total number of steps for training
            tb_log_name: Name for TensorBoard logs
            **kwargs: Additional parameters for learn()
        """
        # Ensure vec_env exists for training
        if self.vec_env is None and hasattr(self.model, 'env'):
            # If vec_env not created, use env from model
            pass
        elif self.vec_env is None:
            # Create vec_env if it doesn't exist
            def make_env():
                return self.env
            self.vec_env = DummyVecEnv([make_env])

        self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name or f"agent_{self.agent_id}",
            **kwargs,
        )

    def save(self, path: str):
        """
        Save model.

        Args:
            path: Path to save model
        """
        self.model.save(path)

    def load(self, path: str):
        """
        Load model.

        Args:
            path: Path to saved model
        """
        self.model = PPO.load(path, env=self.env)

    def reset(self):
        """Reset agent state."""
        # PPO has no internal state that needs resetting
        pass
