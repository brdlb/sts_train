"""
Vectorized environment wrapper for Perudo with multiple tables (self-play).
Each environment represents one table with 4 agents.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple, Any, TYPE_CHECKING
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from .perudo_env import PerudoEnv
from stable_baselines3 import PPO

# Lazy import to avoid circular dependency
try:
    from ..training.opponent_pool import OpponentPool
except ImportError:
    OpponentPool = None  # type: ignore


class PerudoMultiAgentVecEnv(VecEnv):
    """
    Vectorized environment for Perudo with multiple tables.

    Each environment represents one table with 4 agents.
    One agent (agent_id=0) is the learning agent, others are opponents from pool.
    """

    def __init__(
        self,
        num_envs: int,
        num_players: int = 4,
        dice_per_player: int = 5,
        total_dice_values: int = 6,
        max_quantity: int = 30,
        history_length: int = 10,
        opponent_pool: Optional[Any] = None,  # OpponentPool type
        current_model: Optional[PPO] = None,
    ):
        """
        Initialize vectorized environment.

        Args:
            num_envs: Number of parallel environments (tables)
            num_players: Number of players per table
            dice_per_player: Number of dice per player
            total_dice_values: Total possible dice values (usually 6)
            max_quantity: Maximum dice quantity in bid
            history_length: Bid history length in observation
            opponent_pool: Opponent pool for sampling opponents
            current_model: Current learning model (for self-play)
        """
        self.num_envs = num_envs
        self.num_players = num_players
        self.opponent_pool = opponent_pool
        self.current_model = current_model

        # Create environments
        self.envs: List[PerudoEnv] = []
        for i in range(num_envs):
            env = PerudoEnv(
                num_players=num_players,
                dice_per_player=dice_per_player,
                total_dice_values=total_dice_values,
                max_quantity=max_quantity,
                history_length=history_length,
            )
            self.envs.append(env)

        # Opponent models for each environment
        # Each env has num_players-1 opponent models (agent 0 is learning agent)
        self.opponent_models: List[List[Optional[PPO]]] = [
            [None] * (num_players - 1) for _ in range(num_envs)
        ]

        # Opponent paths for each environment (to track which snapshot was used)
        self.opponent_paths: List[List[Optional[str]]] = [
            [None] * (num_players - 1) for _ in range(num_envs)
        ]

        # Current player for each environment
        self.current_players: List[int] = [0] * num_envs

        # Track which agent is active in each environment
        self.active_agent_ids: List[int] = [0] * num_envs

        # Track episode info for each environment
        self.episode_info: List[Dict] = [{}] * num_envs

        # Current training step (for opponent sampling)
        self.current_step: int = 0

        # Get observation and action spaces from first environment
        obs_space = self.envs[0].observation_space
        action_space = self.envs[0].action_space

        super().__init__(num_envs, obs_space, action_space)

    def reset(self, seeds: Optional[List[int]] = None, options: Optional[List[Dict]] = None, current_step: Optional[int] = None) -> np.ndarray:
        """
        Reset all environments.

        Args:
            seeds: Optional seeds for each environment
            options: Optional reset options for each environment
            current_step: Current training step (for opponent sampling).
                          If None, uses self.current_step

        Returns:
            Array of observations for all environments
        """
        if seeds is None:
            seeds = [None] * self.num_envs
        if options is None:
            options = [None] * self.num_envs
        if current_step is None:
            current_step = self.current_step

        observations = []
        for i, env in enumerate(self.envs):
            # Sample opponents from pool for this environment
            if self.opponent_pool is not None:
                self._sample_opponents_for_env(i, current_step)

            # Reset environment
            obs, info = env.reset(seed=seeds[i], options=options[i])
            self.current_players[i] = env.game_state.current_player
            self.active_agent_ids[i] = self.current_players[i]
            self.episode_info[i] = info

            observations.append(obs)

        return np.array(observations)

    def _sample_opponents_for_env(self, env_idx: int, current_step: int = 0):
        """Sample opponents from pool for a specific environment."""
        if self.opponent_pool is None:
            return

        # For each opponent slot (agents 1, 2, 3)
        for opp_idx in range(1, self.num_players):
            # Sample opponent from pool
            opponent_path = self.opponent_pool.sample_opponent(current_step)

            if opponent_path and self.envs[env_idx] is not None:
                # Load opponent model
                opponent_model = self.opponent_pool.load_snapshot(
                    opponent_path, self.envs[env_idx]
                )
                self.opponent_models[env_idx][opp_idx - 1] = opponent_model
                self.opponent_paths[env_idx][opp_idx - 1] = opponent_path
            else:
                # Use current model for self-play
                self.opponent_models[env_idx][opp_idx - 1] = self.current_model
                self.opponent_paths[env_idx][opp_idx - 1] = None

    def step_async(self, actions: np.ndarray) -> None:
        """
        Step all environments asynchronously.

        Args:
            actions: Array of actions for all environments
        """
        self._actions = actions

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Wait for all environments to finish stepping.

        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        observations = []
        rewards = []
        dones = []
        infos = []

        for i, env in enumerate(self.envs):
            # Process steps until it's the learning agent's turn
            # or until the episode ends
            done = False
            reward = 0.0
            obs = None
            info = {}

            while not done and self.active_agent_ids[i] != 0:
                # Opponent's turn
                opponent_idx = self.active_agent_ids[i] - 1  # Convert to opponent index
                opponent_model = self.opponent_models[i][opponent_idx]

                if opponent_model is not None:
                    # Get observation for opponent
                    obs_for_opp = env.get_observation_for_player(self.active_agent_ids[i])
                    obs_array = obs_for_opp.reshape(1, -1)

                    # Get action from opponent model
                    action, _ = opponent_model.predict(obs_array, deterministic=False)
                    action = int(action[0])

                    # Step environment with opponent action
                    obs_for_opp, opp_reward, terminated, truncated, opp_info = env.step(action)
                    done = terminated or truncated

                    # Update current player
                    self.current_players[i] = env.game_state.current_player
                    self.active_agent_ids[i] = self.current_players[i]

                    # Store info for later use
                    info = opp_info
                else:
                    # No opponent model, skip this step
                    self.active_agent_ids[i] = (self.active_agent_ids[i] + 1) % self.num_players
                    break

            # Now it's the learning agent's turn (or episode ended)
            if not done and self.active_agent_ids[i] == 0:
                # Learning agent's turn
                action = int(self._actions[i])
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Update current player
                self.current_players[i] = env.game_state.current_player
                self.active_agent_ids[i] = self.current_players[i]

                # Check if episode ended
                if done and self.opponent_pool is not None and "winner" in info:
                    winner = info.get("winner")
                    won = winner == 0  # Learning agent is agent 0
                    # Update winrate statistics for opponents
                    for opp_idx in range(1, self.num_players):
                        opponent_path = self.opponent_paths[i][opp_idx - 1]
                        if opponent_path is not None:
                            self.opponent_pool.update_winrate(opponent_path, won)

            # Get observation for learning agent (if not already obtained)
            if obs is None:
                if done:
                    # Episode ended, use last observation
                    obs = env.get_observation_for_player(0)
                else:
                    obs = env.get_observation_for_player(0)

            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)  # info ТЕПЕРЬ СОДЕРЖИТ episode_reward и episode_length из внутреннего env

        return (
            np.array(observations),
            np.array(rewards),
            np.array(dones),
            np.array(dones),  # truncated (same as done for now)
            infos,
        )

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments synchronously.

        Args:
            actions: Array of actions for all environments

        Returns:
            Tuple of (observations, rewards, dones, infos)
        """
        self.step_async(actions)
        return self.step_wait()

    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            env.close()

    def env_is_wrapped(self, wrapper_class) -> List[bool]:
        """Check if environments are wrapped."""
        return [False] * self.num_envs

    def get_attr(self, attr_name: str, indices: Optional[List[int]] = None) -> List:
        """Get attribute from environments."""
        if indices is None:
            indices = list(range(self.num_envs))
        return [getattr(self.envs[i], attr_name) for i in indices]

    def set_attr(self, attr_name: str, value: Any, indices: Optional[List[int]] = None) -> None:
        """Set attribute in environments."""
        if indices is None:
            indices = list(range(self.num_envs))
        for i in indices:
            setattr(self.envs[i], attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: Optional[List[int]] = None,
        **method_kwargs,
    ) -> List:
        """Call method on environments."""
        if indices is None:
            indices = list(range(self.num_envs))
        return [
            getattr(self.envs[i], method_name)(*method_args, **method_kwargs)
            for i in indices
        ]

