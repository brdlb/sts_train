"""
Vectorized environment wrapper for Perudo with multiple tables (self-play).
Each environment represents one table with 4 agents.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple, Any, TYPE_CHECKING
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from .perudo_env import PerudoEnv, get_debug_logger
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
        max_history_length: Optional[int] = None,
        opponent_pool: Optional[Any] = None,  # OpponentPool type
        current_model: Optional[PPO] = None,
    ):
        """
        Initialize vectorized environment.

        Args:
            num_envs: Number of parallel environments (tables)
            num_players: Maximum number of players per table (actual will be 3-8, randomly selected)
            dice_per_player: Number of dice per player
            total_dice_values: Total possible dice values (usually 6)
            max_quantity: Maximum dice quantity in bid
            history_length: Bid history length in observation (deprecated, use max_history_length)
            max_history_length: Maximum length of bid history sequence (defaults to history_length)
            opponent_pool: Opponent pool for sampling opponents
            current_model: Current learning model (for self-play)
        """
        self.num_envs = num_envs
        self.max_num_players = max(8, num_players)  # Maximum 8 players for random selection
        self.num_players = num_players  # Will be updated per environment in reset()
        self.opponent_pool = opponent_pool
        self.current_model = current_model

        # Create environments with maximum number of players
        self.envs: List[PerudoEnv] = []
        for i in range(num_envs):
            env = PerudoEnv(
                num_players=self.max_num_players,  # Use max for observation space
                dice_per_player=dice_per_player,
                total_dice_values=total_dice_values,
                max_quantity=max_quantity,
                history_length=history_length,
                max_history_length=max_history_length,  # Use provided max_history_length
            )
            self.envs.append(env)

        # Opponent models for each environment
        # Initialize with maximum size (7 opponents for 8 players max, agent 0 is learning agent)
        # Actual number of opponents will vary based on actual number of players in each episode
        self.opponent_models: List[List[Optional[PPO]]] = [
            [None] * (self.max_num_players - 1) for _ in range(num_envs)
        ]

        # Opponent paths for each environment (to track which snapshot was used)
        self.opponent_paths: List[List[Optional[str]]] = [
            [None] * (self.max_num_players - 1) for _ in range(num_envs)
        ]

        # Current player for each environment
        self.current_players: List[int] = [0] * num_envs

        # Track which agent is active in each environment
        self.active_agent_ids: List[int] = [0] * num_envs

        # Track episode info for each environment
        self.episode_info: List[Dict] = [{}] * num_envs

        # Track episode statistics for learning agent (agent 0) separately
        # This is needed because episode can end on opponent's turn
        self.learning_agent_episode_reward: List[float] = [0.0] * num_envs
        self.learning_agent_episode_length: List[int] = [0] * num_envs

        # Track if episode was already done in previous step (to prevent infinite loops)
        self.episode_already_done: List[bool] = [False] * num_envs

        # Current training step (for opponent sampling)
        self.current_step: int = 0

        # Get observation and action spaces from first environment
        obs_space = self.envs[0].observation_space
        action_space = self.envs[0].action_space

        super().__init__(num_envs, obs_space, action_space)

    def reset(self, seeds: Optional[List[int]] = None, options: Optional[List[Dict]] = None, current_step: Optional[int] = None):
        """Reset all environments."""
        if seeds is None:
            seeds = [None] * self.num_envs
        if options is None:
            options = [None] * self.num_envs
        if current_step is None:
            current_step = self.current_step

        # For Dict observations, we need to collect observations as dicts and then convert to dict of arrays
        observations_list = []
        for i, env in enumerate(self.envs):
            # Reset environment (this will randomly select number of players 3-8)
            obs, info = env.reset(seed=seeds[i], options=options[i])
            
            # Get actual number of players for this episode
            actual_num_players = env.num_players
            
            # Extract opponent snapshot IDs from options if provided
            opponent_snapshot_ids = None
            if options[i] is not None and "opponent_snapshot_ids" in options[i]:
                opponent_snapshot_ids = options[i]["opponent_snapshot_ids"]
            
            # Sample opponents from pool for this environment
            # Number of opponents = actual_num_players - 1 (agent 0 is learning agent)
            if self.opponent_pool is not None:
                self._sample_opponents_for_env(
                    i, current_step, actual_num_players, opponent_snapshot_ids
                )
            
            self.current_players[i] = env.game_state.current_player
            self.active_agent_ids[i] = self.current_players[i]
            self.episode_info[i] = info
            # Reset learning agent episode statistics
            self.learning_agent_episode_reward[i] = 0.0
            self.learning_agent_episode_length[i] = 0
            # Reset episode done flag
            self.episode_already_done[i] = False

            # ===== НОВОЕ: ЕСЛИ ПЕРВЫЙ ХОД НЕ LEARNING AGENT, ПРОПУСКАЕМ ДО НЕГО =====
            max_steps = 100
            steps = 0
            actual_num_players = env.num_players  # Get actual number of players for this episode
            while self.active_agent_ids[i] != 0 and steps < max_steps:
                steps += 1
                
                # Opponent's turn
                opponent_idx = self.active_agent_ids[i] - 1
                # Only use opponent model if opponent_idx is within bounds
                if opponent_idx < len(self.opponent_models[i]):
                    opponent_model = self.opponent_models[i][opponent_idx]
                else:
                    opponent_model = None

                if opponent_model is not None:
                    obs_for_opp = env.get_observation_for_player(self.active_agent_ids[i])
                    
                    # Handle both dict and array observations
                    if isinstance(obs_for_opp, dict):
                        obs_for_predict = {key: np.array([value]) for key, value in obs_for_opp.items()}
                    else:
                        obs_for_predict = obs_for_opp.reshape(1, -1)
                    
                    action, _ = opponent_model.predict(obs_for_predict, deterministic=False)
                    action = int(action[0])
                    
                    env.set_active_player(self.active_agent_ids[i])
                    obs_for_opp, _, terminated, opp_truncated, opp_info = env.step(action)
                    
                    if terminated or opp_truncated:
                        # Если игра завершилась до хода learning agent, информация об эпизоде 
                        # уже сохранена в env.last_episode_reward и env.last_episode_length
                        # Но VecMonitor не увидит этот эпизод, потому что он завершился в reset()
                        # Это допустимо - такие эпизоды не должны учитываться, так как learning agent
                        # не делал в них ходов
                        # Сбросить снова
                        obs, info = env.reset(seed=seeds[i], options=options[i])
                        self.active_agent_ids[i] = env.game_state.current_player
                        steps = 0  # Начать заново
                        continue
                    
                    self.current_players[i] = env.game_state.current_player
                    self.active_agent_ids[i] = self.current_players[i]
                else:
                    break

            # Get observation for learning agent
            obs = env.get_observation_for_player(0)
            observations_list.append(obs)

        # Convert list of dicts to dict of arrays for VecEnv
        if len(observations_list) == 0:
            # Should not happen, but handle gracefully
            return observations_list
        
        if isinstance(observations_list[0], dict):
            # Dict observation space: convert to dict of arrays
            observations_dict = {
                key: np.array([obs[key] for obs in observations_list])
                for key in observations_list[0].keys()
            }
            return observations_dict
        else:
            # Array observation space: convert to array
            return np.array(observations_list)

    def _sample_opponents_for_env(
        self,
        env_idx: int,
        current_step: int = 0,
        num_players: Optional[int] = None,
        opponent_snapshot_ids: Optional[List[Optional[str]]] = None,
    ):
        """
        Sample opponents from pool for a specific environment.
        
        Args:
            env_idx: Environment index
            current_step: Current training step
            num_players: Actual number of players for this episode (if None, uses max)
            opponent_snapshot_ids: Optional list of snapshot IDs to use for each opponent.
                                 If provided, must have length >= num_players - 1.
                                 Use None in list for random sampling for that opponent.
        """
        if self.opponent_pool is None:
            return

        # Use actual number of players if provided, otherwise use max
        if num_players is None:
            num_players = self.max_num_players

        # For each opponent slot (agents 1, 2, ..., num_players-1)
        for opp_idx in range(1, num_players):
            # Get snapshot ID if specified
            snapshot_id = None
            if opponent_snapshot_ids is not None and opp_idx - 1 < len(opponent_snapshot_ids):
                snapshot_id = opponent_snapshot_ids[opp_idx - 1]

            # Sample opponent from pool (with optional specific snapshot ID)
            opponent_path = self.opponent_pool.sample_opponent(current_step, snapshot_id=snapshot_id)

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

    def step_wait(self):
        """
        Wait for all environments to finish stepping.

        Returns:
            Tuple of (observations, rewards, dones, infos)
            Note: observations can be dict or array depending on observation_space
            Note: dones combines both terminated and truncated for VecMonitor compatibility
        """
        observations_list = []
        rewards = []
        dones = []
        infos = []

        for i, env in enumerate(self.envs):
            done = False
            reward = 0.0
            obs = None
            info = {}

            # ===== ПРОВЕРКА: ЕСЛИ ЭПИЗОД УЖЕ ЗАВЕРШЕН, АВТОМАТИЧЕСКИ ВЫЗВАТЬ RESET =====
            if self.episode_already_done[i]:
                # Episode was already done, automatically reset to start new episode
                # This prevents infinite loops where done=True is returned repeatedly
                
                # Reset environment (this will randomly select number of players 3-8)
                obs, info = env.reset()
                
                # Get actual number of players for this episode
                actual_num_players = env.num_players
                
                # Sample opponents from pool for this environment
                if self.opponent_pool is not None:
                    self._sample_opponents_for_env(i, self.current_step, actual_num_players)
                
                self.current_players[i] = env.game_state.current_player
                self.active_agent_ids[i] = self.current_players[i]
                self.episode_info[i] = info
                # Reset learning agent episode statistics
                self.learning_agent_episode_reward[i] = 0.0
                self.learning_agent_episode_length[i] = 0
                # Reset episode done flag
                self.episode_already_done[i] = False
                
                # If first turn is not learning agent, advance to learning agent's turn
                max_steps = 100
                steps = 0
                actual_num_players = env.num_players  # Get actual number of players for this episode
                while self.active_agent_ids[i] != 0 and steps < max_steps:
                    steps += 1
                    
                    # Opponent's turn
                    opponent_idx = self.active_agent_ids[i] - 1
                    # Only use opponent model if opponent_idx is within bounds
                    if opponent_idx < len(self.opponent_models[i]):
                        opponent_model = self.opponent_models[i][opponent_idx]
                    else:
                        opponent_model = None
                    
                    if opponent_model is not None:
                        obs_for_opp = env.get_observation_for_player(self.active_agent_ids[i])
                        obs_array = obs_for_opp.reshape(1, -1)
                        
                        action, _ = opponent_model.predict(obs_array, deterministic=False)
                        action = int(action[0])
                        
                        env.set_active_player(self.active_agent_ids[i])
                        obs_for_opp, _, terminated, opp_truncated, opp_info = env.step(action)
                        
                        if terminated or opp_truncated:
                            # If game ended before learning agent's turn, reset again
                            obs, info = env.reset()
                            # Re-sample opponents after reset
                            if self.opponent_pool is not None:
                                self._sample_opponents_for_env(i, self.current_step, env.num_players)
                            self.active_agent_ids[i] = env.game_state.current_player
                            self.current_players[i] = env.game_state.current_player
                            # Reset learning agent episode statistics
                            self.learning_agent_episode_reward[i] = 0.0
                            self.learning_agent_episode_length[i] = 0
                            steps = 0  # Start over
                            continue
                        
                        self.current_players[i] = env.game_state.current_player
                        self.active_agent_ids[i] = self.current_players[i]
                    else:
                        # No opponent model available, skip to next player
                        self.active_agent_ids[i] = (self.active_agent_ids[i] + 1) % actual_num_players
                        break
                
                # Get observation for learning agent
                obs = env.get_observation_for_player(0)
                done = False
                reward = 0.0
                info = {}
                
                observations.append(obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
                continue

            # ===== ШАГ 1: LEARNING AGENT ДЕЛАЕТ ХОД =====
            action = int(self._actions[i])
            env.set_active_player(0)

            obs, reward, terminated, truncated_flag, info = env.step(action)
            done = terminated or truncated_flag

            # Track learning agent's reward and length (only if episode not already done)
            if not done:
                self.learning_agent_episode_reward[i] += reward
                self.learning_agent_episode_length[i] += 1
            else:
                # Episode just ended, accumulate this step's reward but mark as done
                self.learning_agent_episode_reward[i] += reward
                self.learning_agent_episode_length[i] += 1
                self.episode_already_done[i] = True

            # Update current player
            self.current_players[i] = env.game_state.current_player
            self.active_agent_ids[i] = self.current_players[i]

            # ===== ШАГ 2: ТЕПЕРЬ ОППОНЕНТЫ ХОДЯТ, ПОКА СНОВА НЕ НАСТАНЕТ ХОД LEARNING AGENT =====
            opponent_step_count = 0
            max_opponent_steps = 100  # Защита от бесконечного цикла
            actual_num_players = env.num_players  # Get actual number of players for this episode

            while not done and self.active_agent_ids[i] != 0 and opponent_step_count < max_opponent_steps:
                opponent_step_count += 1

                # Opponent's turn
                opponent_idx = self.active_agent_ids[i] - 1
                # Only use opponent model if opponent_idx is within bounds
                if opponent_idx < len(self.opponent_models[i]):
                    opponent_model = self.opponent_models[i][opponent_idx]
                else:
                    opponent_model = None

                if opponent_model is not None:
                    # Get observation for opponent
                    obs_for_opp = env.get_observation_for_player(self.active_agent_ids[i])
                    
                    # Handle both dict and array observations
                    if isinstance(obs_for_opp, dict):
                        obs_for_predict = {key: np.array([value]) for key, value in obs_for_opp.items()}
                    else:
                        obs_for_predict = obs_for_opp.reshape(1, -1)
                    
                    # Get action from opponent model
                    action, _ = opponent_model.predict(obs_for_predict, deterministic=False)
                    action = int(action[0])

                    env.set_active_player(self.active_agent_ids[i])

                    # Step environment with opponent action
                    obs_for_opp, opp_reward, terminated, opp_truncated, opp_info = env.step(action)
                    done = terminated or opp_truncated

                    # If episode ended on opponent's turn, mark as done to prevent infinite loops
                    if done:
                        self.episode_already_done[i] = True

                    # Update current player
                    self.current_players[i] = env.game_state.current_player
                    self.active_agent_ids[i] = self.current_players[i]

                    # Store info for later use
                    # If episode ended on opponent's turn, opp_info will contain "episode" key
                    info = opp_info
                else:
                    # No opponent model, skip this step
                    # Use actual number of players for this episode
                    self.active_agent_ids[i] = (self.active_agent_ids[i] + 1) % actual_num_players
                    break

            # ===== ШАГ 3: ПРОВЕРКА ЗАВЕРШЕНИЯ И ОБНОВЛЕНИЕ СТАТИСТИКИ =====
            if done and self.opponent_pool is not None and "winner" in info:
                winner = info.get("winner")
                won = winner == 0  # Learning agent is agent 0
                # Update winrate statistics for opponents
                # Use actual number of players for this episode
                actual_num_players = env.num_players
                for opp_idx in range(1, actual_num_players):
                    if opp_idx - 1 < len(self.opponent_paths[i]):
                        opponent_path = self.opponent_paths[i][opp_idx - 1]
                        if opponent_path is not None:
                            self.opponent_pool.update_winrate(opponent_path, won)

            # CRITICAL: VecMonitor expects info["episode"] to exist when done=True
            # We MUST use learning agent's statistics, not opponent's or environment's
            # This is because episode can end on opponent's turn, but we need stats for learning agent
            if done:
                # Always use learning agent's tracked statistics
                # These are accumulated only when learning agent (agent 0) makes moves
                learning_reward = self.learning_agent_episode_reward[i]
                learning_length = self.learning_agent_episode_length[i]
                
                # Only mark as done if episode just ended (not already done)
                if not self.episode_already_done[i]:
                    self.episode_already_done[i] = True
                
                # Get episode statistics from environment (bid_count, challenge_count, believe_count, winner)
                # These are tracked for the entire episode, not just learning agent
                bid_count = getattr(env, 'episode_bid_count', 0)
                challenge_count = getattr(env, 'episode_challenge_count', 0)
                believe_count = getattr(env, 'episode_believe_count', 0)
                winner = env.game_state.winner if hasattr(env.game_state, "winner") and env.game_state.winner is not None else -1
                
                # Override episode info with learning agent's statistics and episode statistics
                info["episode"] = {
                    "r": learning_reward,
                    "l": learning_length,
                    "bid_count": bid_count,
                    "challenge_count": challenge_count,
                    "believe_count": believe_count,
                    "winner": winner,
                }
                info["episode_reward"] = learning_reward
                info["episode_length"] = learning_length
                
                # Keep winner info if available
                if "winner" not in info and hasattr(env.game_state, "winner"):
                    info["winner"] = env.game_state.winner

            # Get observation for learning agent
            if obs is None or self.active_agent_ids[i] != 0:
                obs = env.get_observation_for_player(0)

            observations_list.append(obs)
            rewards.append(reward)
            # Combine terminated and truncated into dones for VecMonitor compatibility
            dones.append(done)
            infos.append(info)

        # CRITICAL: Ensure we have exactly num_envs results
        # This prevents ValueError when VecMonitor tries to broadcast arrays
        if len(observations_list) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} results but got {len(observations_list)}. "
                f"This indicates a bug in step_wait() where not all environments "
                f"added results to the output lists."
            )

        # Convert observations to proper format for VecEnv
        if len(observations_list) > 0 and isinstance(observations_list[0], dict):
            # Dict observation space: convert to dict of arrays
            observations = {
                key: np.array([obs[key] for obs in observations_list])
                for key in observations_list[0].keys()
            }
        else:
            # Array observation space: convert to array
            observations = np.array(observations_list)

        return (
            observations,
            np.array(rewards),
            np.array(dones),
            infos,
        )

    def step(self, actions: np.ndarray):
        """
        Step all environments synchronously.

        Args:
            actions: Array of actions for all environments

        Returns:
            Tuple of (observations, rewards, dones, infos)
            Note: observations can be dict or array depending on observation_space
            Note: dones combines both terminated and truncated for VecMonitor compatibility
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

