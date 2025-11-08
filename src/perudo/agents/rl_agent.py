"""
RL agent based on Stable Baselines3.
"""

import numpy as np
import contextlib
from io import StringIO
from typing import Optional
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from .base_agent import BaseAgent
from ..game.perudo_env import PerudoEnv


def get_device(device: Optional[str] = None) -> str:
    """
    Get device for training with automatic GPU detection and CPU fallback.
    
    Args:
        device: Device string (e.g., "cpu", "cuda", "cuda:0"). 
                If None, automatically detects GPU and falls back to CPU.
    
    Returns:
        Device string to use for training
    """
    if device is not None:
        return device
    
    # Try to use GPU if available
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    return device


class RLAgent(BaseAgent):
    """RL agent using Stable Baselines3."""

    def __init__(
        self,
        agent_id: int,
        env: PerudoEnv,
        model: Optional[MaskablePPO] = None,
        policy: str = "MlpPolicy",
        device: Optional[str] = None,  # If None, auto-detects GPU with CPU fallback
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
        load_model_later: bool = False,
    ):
        """
        Initialize RL agent.

        Args:
            agent_id: Unique agent ID
            env: Perudo environment
            model: Existing MaskablePPO model (if any)
            policy: Policy type (only used if model is None and load_model_later is False)
            device: Device string (e.g., "cpu", "cuda"). If None, auto-detects GPU with CPU fallback
            learning_rate: Learning rate (only used if model is None and load_model_later is False)
            n_steps: Number of steps to collect data before update (only used if model is None and load_model_later is False)
            batch_size: Batch size for training (only used if model is None and load_model_later is False)
            n_epochs: Number of epochs to update on one data collection (only used if model is None and load_model_later is False)
            gamma: Discount factor (only used if model is None and load_model_later is False)
            gae_lambda: GAE parameter (Generalized Advantage Estimation) (only used if model is None and load_model_later is False)
            clip_range: Clipping parameter for PPO (only used if model is None and load_model_later is False)
            ent_coef: Entropy coefficient (for exploration) (only used if model is None and load_model_later is False)
            vf_coef: Value function coefficient (only used if model is None and load_model_later is False)
            max_grad_norm: Maximum gradient norm (only used if model is None and load_model_later is False)
            verbose: Verbosity level
            load_model_later: If True, don't create model now (will be loaded via load() method)
        """
        super().__init__(agent_id)

        self.env = env
        self.verbose = verbose

        if model is not None:
            self.model = model
            self.vec_env = None  # If model already created, vec_env not needed
        elif load_model_later:
            # Don't create model now - it will be loaded via load() method
            # This is useful for server-side loading where we want to load
            # a model with specific architecture (e.g., MultiInputPolicy + TransformerFeaturesExtractor)
            self.model = None
            self.vec_env = None
        else:
            # Create environment wrapper for SB3
            def make_env():
                return env

            self.vec_env = DummyVecEnv([make_env])

            # Determine device for training (GPU with CPU fallback)
            device_str = get_device(device)

            # Create MaskablePPO model
            self.model = MaskablePPO(
                policy=policy,
                env=self.vec_env,
                device=device_str,
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

    def act(self, observation, deterministic: bool = False) -> int:
        """
        Choose action based on observation.

        Args:
            observation: Observation from environment (can be np.ndarray or Dict)
            deterministic: Whether to use deterministic policy

        Returns:
            Action from action_space
        
        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() method first.")
        
        # Convert observation to format for SB3
        # For Dict observations, SB3 expects a dict with batched arrays
        # For array observations, add batch dimension
        if isinstance(observation, dict):
            # Dict observation: batch each key
            obs_dict = {
                key: np.array([value]) if isinstance(value, np.ndarray) else [value]
                for key, value in observation.items()
            }
            action, _ = self.model.predict(obs_dict, deterministic=deterministic)
        else:
            # Array observation: add batch dimension
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
        
        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Cannot train.")
        
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
        
        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Cannot save.")
        self.model.save(path)

    def load(self, path: str, device: Optional[str] = "cpu"):
        """
        Load model from path.
        
        This method loads a saved MaskablePPO model. The model architecture
        (policy type, features extractor, etc.) is automatically restored from
        the saved file. This is important for models trained with custom
        architectures like MultiInputPolicy with TransformerFeaturesExtractor.

        Args:
            path: Path to saved model file (.zip)
            device: Device to load model on (default: "cpu" for server usage).
                   Use "cpu" for inference to avoid GPU overhead and memory issues.
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model cannot be loaded or is incompatible with environment
        """
        import os
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        # Suppress SB3 wrapping messages
        with contextlib.redirect_stdout(StringIO()):
            try:
                # Load model with specified device
                # SB3 will automatically restore the policy architecture (e.g., MultiInputPolicy)
                # and features extractor (e.g., TransformerFeaturesExtractor) from the saved file
                # The env parameter is used for validation (observation_space, action_space)
                self.model = MaskablePPO.load(path, env=self.env, device=device)
            except Exception as e:
                raise ValueError(
                    f"Failed to load model from {path}: {str(e)}\n"
                    f"Make sure the model was trained with the same environment configuration "
                    f"(observation_space, action_space, max_history_length, etc.)"
                ) from e

    def reset(self):
        """Reset agent state."""
        # PPO has no internal state that needs resetting
        pass
