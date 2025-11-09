"""
Vectorized environment wrapper for Perudo with multiple tables (self-play).
Each environment represents one table with 4 agents.
"""

import random
import numpy as np
from typing import List, Optional, Dict, Tuple, Any, TYPE_CHECKING
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from .perudo_env import PerudoEnv

# Lazy import to avoid circular dependency
if TYPE_CHECKING:
    from sb3_contrib import MaskablePPO
    PPO = MaskablePPO  # Alias for type hints
else:
    PPO = Any  # Runtime fallback

try:
    from ..training.opponent_pool import OpponentPool
except ImportError:
    OpponentPool = None  # type: ignore

try:
    from ..training.config import RewardConfig, DEFAULT_CONFIG
except ImportError:
    RewardConfig = None  # type: ignore
    DEFAULT_CONFIG = None  # type: ignore

try:
    from ..utils.helpers import calculate_reward
except ImportError:
    calculate_reward = None  # type: ignore


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
        random_num_players: bool = True,
        min_players: int = 3,
        max_players: int = 8,
        reward_config: Optional[Any] = None,  # RewardConfig type
        debug_moves: bool = False,
    ):
        """
        Initialize vectorized environment.

        Args:
            num_envs: Number of parallel environments (tables)
            num_players: Number of players per table (used if random_num_players=False, or as max for observation space)
            dice_per_player: Number of dice per player
            total_dice_values: Total possible dice values (usually 6)
            max_quantity: Maximum dice quantity in bid
            history_length: Bid history length in observation (deprecated, use max_history_length)
            max_history_length: Maximum length of bid history sequence (defaults to history_length)
            opponent_pool: Opponent pool for sampling opponents
            current_model: Current learning model (for self-play)
            random_num_players: If True, randomly select num_players in each episode
            min_players: Minimum number of players (used when random_num_players=True)
            max_players: Maximum number of players (used when random_num_players=True)
            reward_config: Reward configuration (uses DEFAULT_CONFIG.reward if not provided)
            debug_moves: Enable detailed move logging for debugging
        """
        self.num_envs = num_envs
        self.random_num_players = random_num_players
        self.min_players = min_players
        self.max_players = max_players
        
        # Use maximum number of players for observation space
        if random_num_players:
            self.max_num_players = max(max_players, num_players)  # At least max_players for random selection
        else:
            self.max_num_players = num_players
        
        self.num_players = num_players  # Will be updated per environment in reset() if random_num_players=True
        self.opponent_pool = opponent_pool
        self.current_model = current_model

        # Use default reward config if not provided
        if reward_config is None and DEFAULT_CONFIG is not None:
            reward_config = DEFAULT_CONFIG.reward

        # Create environments with maximum number of players
        self.envs: List[PerudoEnv] = []
        for i in range(num_envs):
            env = PerudoEnv(
                num_players=self.max_num_players,  # Use max for observation space
                dice_per_player=dice_per_player,
                total_dice_values=total_dice_values,
                max_quantity=max_quantity,
                history_length=history_length,
                random_num_players=random_num_players,
                min_players=min_players,
                max_players=max_players,
                max_history_length=max_history_length,  # Use provided max_history_length
                reward_config=reward_config,
                debug_moves=debug_moves,
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
        
        # Track accumulated reward for VecMonitor
        # VecMonitor accumulates rewards from step_reward, and we need to know what it has accumulated
        # so we can return the correct difference when episode ends
        self.vecmonitor_accumulated_reward: List[float] = [0.0] * num_envs

        # Track episode statistics for all learning agents when pool is empty
        # Format: List[Dict[agent_id, reward/length]] for each environment
        self.all_agents_episode_reward: List[Dict[int, float]] = [
            {} for _ in range(num_envs)
        ]
        self.all_agents_episode_length: List[Dict[int, int]] = [
            {} for _ in range(num_envs)
        ]

        # Track if episode was already done in previous step (to prevent infinite loops)
        self.episode_already_done: List[bool] = [False] * num_envs

        # Track if all agents should learn (when opponent pool is empty)
        self.all_agents_learn_mode: List[bool] = [False] * num_envs

        # Current training step (for opponent sampling)
        self.current_step: int = 0

        # Store last observations for action_masks() method (required by MaskablePPO)
        self.last_obs: Optional[Dict[str, np.ndarray]] = None

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
            # Reset all agents episode statistics
            self.all_agents_episode_reward[i] = {}
            self.all_agents_episode_length[i] = {}
            # Reset episode done flag
            self.episode_already_done[i] = False
            # Reset VecMonitor accumulated reward tracker
            self.vecmonitor_accumulated_reward[i] = 0.0

            # ===== НОВОЕ: ЕСЛИ ПЕРВЫЙ ХОД НЕ LEARNING AGENT, ПРОПУСКАЕМ ДО НЕГО =====
            # Skip only if not in all_learn mode (in all_learn mode, all agents are learning)
            if not self.all_agents_learn_mode[i]:
                max_steps = 100
                steps = 0
                actual_num_players = env.num_players  # Get actual number of players for this episode
                while self.active_agent_ids[i] != 0 and steps < max_steps:
                    steps += 1
                    
                    # CRITICAL: Skip players with no dice - they have already lost and cannot make moves
                    # Use next_player() which automatically skips players with 0 dice
                    if env.game_state.player_dice_count[self.active_agent_ids[i]] == 0:
                        env.game_state.next_player()
                        self.active_agent_ids[i] = env.game_state.current_player
                        self.current_players[i] = env.game_state.current_player
                        # Check if game ended after skipping players
                        if env.game_state.game_over:
                            # Game ended, reset again
                            obs, info = env.reset(seed=seeds[i], options=options[i])
                            self.active_agent_ids[i] = env.game_state.current_player
                            self.current_players[i] = env.game_state.current_player
                            steps = 0  # Start over
                            continue
                        continue
                    
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
                            
                            # Extract action mask if available and pass it to predict()
                            action_masks = None
                            if "action_mask" in obs_for_opp:
                                mask = obs_for_opp["action_mask"]
                                if isinstance(mask, np.ndarray):
                                    mask = mask.astype(bool)
                                    if mask.ndim > 1:
                                        mask = mask.flatten()
                                    action_masks = np.array([mask])
                                elif isinstance(mask, (list, tuple)):
                                    mask = np.array(mask, dtype=bool)
                                    if mask.ndim > 1:
                                        mask = mask.flatten()
                                    action_masks = np.array([mask])
                            
                            # Predict with action mask if available
                            if action_masks is not None:
                                action, _ = opponent_model.predict(obs_for_predict, deterministic=False, action_masks=action_masks)
                            else:
                                action, _ = opponent_model.predict(obs_for_predict, deterministic=False)
                        else:
                            obs_for_predict = obs_for_opp.reshape(1, -1)
                            action, _ = opponent_model.predict(obs_for_predict, deterministic=False)
                        action = int(action[0])
                        
                        env.set_active_player(self.active_agent_ids[i])
                        obs_for_opp, _, terminated, opp_truncated, opp_info = env.step(action)
                        
                        if terminated or opp_truncated:
                            # Если игра завершилась до хода learning agent, эпизод будет обработан
                            # в следующем вызове step_wait() когда learning agent попытается сделать ход
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
            # In all_learn mode, use current active agent, otherwise use agent 0
            if self.all_agents_learn_mode[i]:
                obs = env.get_observation_for_player(self.active_agent_ids[i])
            else:
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
            # Store observations for action_masks() method
            self.last_obs = observations_dict
            return observations_dict
        else:
            # Array observation space: convert to array
            observations_array = np.array(observations_list)
            # Store observations for action_masks() method
            self.last_obs = observations_array
            return observations_array

    def _sample_opponents_for_env(
        self,
        env_idx: int,
        current_step: int = 0,
        num_players: Optional[int] = None,
        opponent_snapshot_ids: Optional[List[Optional[str]]] = None,
    ):
        """
        Sample opponents from pool for a specific environment.
        
        If snapshots exist in the pool, only unique snapshots are added to the room.
        Remaining opponent slots use the current model (self-play), just like in the first iteration.
        
        Args:
            env_idx: Environment index
            current_step: Current training step
            num_players: Actual number of players for this episode (if None, uses max)
            opponent_snapshot_ids: Optional list of snapshot IDs to use for each opponent.
                                 If provided, must have length >= num_players - 1.
                                 Use None in list for random sampling for that opponent.
        """
        if self.opponent_pool is None:
            # If no opponent pool, all agents learn
            self.all_agents_learn_mode[env_idx] = True
            return

        # Use actual number of players if provided, otherwise use max
        if num_players is None:
            num_players = self.max_num_players

        # Number of opponent slots (agents 1, 2, ..., num_players-1)
        num_opponents = num_players - 1

        # Check if pool has snapshots
        if not self.opponent_pool.snapshots:
            # Pool is empty, all opponents use current model (self-play)
            for opp_idx in range(1, num_players):
                self.opponent_models[env_idx][opp_idx - 1] = self.current_model
                self.opponent_paths[env_idx][opp_idx - 1] = None
            self.all_agents_learn_mode[env_idx] = True
            return

        # Get list of available unique snapshot IDs from pool
        available_snapshot_ids = list(self.opponent_pool.snapshots.keys())
        
        # If specific snapshot IDs are provided, use them (but still ensure uniqueness)
        if opponent_snapshot_ids is not None:
            # Filter out None values and ensure uniqueness
            unique_snapshot_ids = []
            seen_ids = set()
            for snap_id in opponent_snapshot_ids:
                if snap_id is not None and snap_id not in seen_ids and snap_id in available_snapshot_ids:
                    unique_snapshot_ids.append(snap_id)
                    seen_ids.add(snap_id)
        else:
            # Sample unique snapshots from pool (without replacement)
            # Shuffle to get random selection
            shuffled_ids = available_snapshot_ids.copy()
            random.shuffle(shuffled_ids)
            # Take at most num_opponents unique snapshots
            unique_snapshot_ids = shuffled_ids[:num_opponents]

        # Assign unique snapshots to opponent slots
        # First, assign unique snapshots to available slots
        num_unique_snapshots = len(unique_snapshot_ids)
        for slot_idx, snapshot_id in enumerate(unique_snapshot_ids):
            if slot_idx >= num_opponents:
                break  # No more opponent slots
            
            opponent_path = self.opponent_pool.get_snapshot_by_id(snapshot_id)
            if opponent_path and self.envs[env_idx] is not None:
                # Load opponent model
                try:
                    opponent_model = self.opponent_pool.load_snapshot(
                        opponent_path, self.envs[env_idx]
                    )
                    if opponent_model is not None:
                        # opp_idx = slot_idx + 1 (because agents are 1-indexed)
                        self.opponent_models[env_idx][slot_idx] = opponent_model
                        self.opponent_paths[env_idx][slot_idx] = opponent_path
                        # Update last used timestamp
                        if snapshot_id in self.opponent_pool.snapshots:
                            self.opponent_pool.snapshots[snapshot_id].last_used = current_step
                    else:
                        # Failed to load snapshot, use current model instead
                        self.opponent_models[env_idx][slot_idx] = self.current_model
                        self.opponent_paths[env_idx][slot_idx] = None
                except Exception as e:
                    # Failed to load snapshot, use current model instead
                    print(f"Warning: Failed to load opponent snapshot {opponent_path}: {e}")
                    self.opponent_models[env_idx][slot_idx] = self.current_model
                    self.opponent_paths[env_idx][slot_idx] = None
            else:
                # No snapshot path found, use current model
                self.opponent_models[env_idx][slot_idx] = self.current_model
                self.opponent_paths[env_idx][slot_idx] = None

        # Fill remaining slots with current model (self-play)
        for slot_idx in range(num_unique_snapshots, num_opponents):
            self.opponent_models[env_idx][slot_idx] = self.current_model
            self.opponent_paths[env_idx][slot_idx] = None

        # If pool had snapshots but we used them all, we still have some snapshots in the room
        # So we don't enable all_agents_learn_mode
        # But if all slots use current_model, enable all_agents_learn_mode
        all_use_current_model = all(
            self.opponent_models[env_idx][i] == self.current_model
            for i in range(num_opponents)
        )
        self.all_agents_learn_mode[env_idx] = all_use_current_model

    def step_async(self, actions: np.ndarray) -> None:
        """
        Step all environments asynchronously.

        Args:
            actions: Array of actions for all environments
        """
        self._actions = actions

    def _calculate_final_reward(self, env, learning_agent_id: int, accumulated_reward: float, reward_config: Optional[Any]) -> float:
        """
        Calculate final reward for learning agent at episode end.
        
        Args:
            env: Environment instance
            learning_agent_id: ID of learning agent (usually 0)
            accumulated_reward: Accumulated reward during episode
            reward_config: Reward configuration
            
        Returns:
            Final reward including win/lose bonuses
        """
        if reward_config is None:
            return accumulated_reward
        
        winner = env.game_state.winner if hasattr(env.game_state, "winner") and env.game_state.winner is not None else -1
        
        if winner == learning_agent_id:
            final_reward = accumulated_reward + reward_config.win_reward
            if winner >= 0 and winner < len(env.game_state.player_dice_count):
                winner_dice_count = env.game_state.player_dice_count[winner]
                if winner_dice_count is not None and reward_config.win_dice_bonus > 0:
                    final_reward += reward_config.win_dice_bonus * winner_dice_count
        else:
            final_reward = accumulated_reward + reward_config.lose_penalty
        
        return final_reward

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
                # Reset episode statistics
                self.learning_agent_episode_reward[i] = 0.0
                self.learning_agent_episode_length[i] = 0
                self.all_agents_episode_reward[i] = {}
                self.all_agents_episode_length[i] = {}
                self.episode_already_done[i] = False
                self.vecmonitor_accumulated_reward[i] = 0.0
                
                # If first turn is not learning agent, advance to learning agent's turn
                # Skip only if not in all_learn mode (in all_learn mode, all agents are learning)
                if not self.all_agents_learn_mode[i]:
                    max_steps = 100
                    steps = 0
                    actual_num_players = env.num_players  # Get actual number of players for this episode
                    try:
                        while self.active_agent_ids[i] != 0 and steps < max_steps:
                            steps += 1
                        
                            # Skip players with no dice - they can't make moves
                            if env.game_state.player_dice_count[self.active_agent_ids[i]] == 0:
                                # Use next_player() which automatically skips players with 0 dice
                                env.game_state.next_player()
                                self.active_agent_ids[i] = env.game_state.current_player
                                self.current_players[i] = env.game_state.current_player
                                # Check if game ended after skipping players
                                if env.game_state.game_over:
                                    # Game ended, reset again
                                    obs, info = env.reset()
                                    # Re-sample opponents after reset
                                    if self.opponent_pool is not None:
                                        self._sample_opponents_for_env(i, self.current_step, env.num_players)
                                    self.active_agent_ids[i] = env.game_state.current_player
                                    self.current_players[i] = env.game_state.current_player
                                    # Reset learning agent episode statistics
                                    self.learning_agent_episode_reward[i] = 0.0
                                    self.learning_agent_episode_length[i] = 0
                                    # Reset all agents episode statistics
                                    self.all_agents_episode_reward[i] = {}
                                    self.all_agents_episode_length[i] = {}
                                    # Reset VecMonitor accumulated reward tracker
                                    self.vecmonitor_accumulated_reward[i] = 0.0
                                    steps = 0  # Start over
                                    continue
                                continue
                            
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
                                    
                                    # Extract action mask if available and pass it to predict()
                                    action_masks = None
                                    if "action_mask" in obs_for_opp:
                                        mask = obs_for_opp["action_mask"]
                                        if isinstance(mask, np.ndarray):
                                            mask = mask.astype(bool)
                                            if mask.ndim > 1:
                                                mask = mask.flatten()
                                            action_masks = np.array([mask])
                                        elif isinstance(mask, (list, tuple)):
                                            mask = np.array(mask, dtype=bool)
                                            if mask.ndim > 1:
                                                mask = mask.flatten()
                                            action_masks = np.array([mask])
                                    
                                    # Predict with action mask if available
                                    if action_masks is not None:
                                        action, _ = opponent_model.predict(obs_for_predict, deterministic=False, action_masks=action_masks)
                                    else:
                                        action, _ = opponent_model.predict(obs_for_predict, deterministic=False)
                                else:
                                    obs_for_predict = obs_for_opp.reshape(1, -1)
                                    action, _ = opponent_model.predict(obs_for_predict, deterministic=False)
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
                                    # Reset all agents episode statistics
                                    self.all_agents_episode_reward[i] = {}
                                    self.all_agents_episode_length[i] = {}
                                    # Reset VecMonitor accumulated reward tracker
                                    self.vecmonitor_accumulated_reward[i] = 0.0
                                    steps = 0  # Start over
                                    continue
                                
                                self.current_players[i] = env.game_state.current_player
                                self.active_agent_ids[i] = self.current_players[i]
                            else:
                                # No opponent model available, skip to next player
                                self.active_agent_ids[i] = (self.active_agent_ids[i] + 1) % actual_num_players
                                break
                    except Exception as e:
                        # If any exception occurs, ensure we still get observation and add result
                        # This prevents missing results when exceptions occur in the loop
                        import traceback
                        print(f"Warning: Exception in episode_already_done reset loop for env {i}: {e}")
                        print(traceback.format_exc())
                        # Continue to get observation and add result
                
                # Get observation for learning agent
                # In all_learn mode, use current active agent, otherwise use agent 0
                if self.all_agents_learn_mode[i]:
                    obs = env.get_observation_for_player(self.active_agent_ids[i])
                else:
                    obs = env.get_observation_for_player(0)
                done = False
                reward = 0.0
                info = {}
                
                observations_list.append(obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
                continue

            # ===== ШАГ 1: LEARNING AGENT ДЕЛАЕТ ХОД =====
            # Check if all agents learn mode is enabled (pool is empty)
            all_learn = self.all_agents_learn_mode[i]
            
            # Determine which agent should make a move
            # In normal mode: only agent 0 learns
            # In all_learn mode: current active agent learns
            current_learning_agent = self.active_agent_ids[i] if all_learn else 0
            
            # Check if learning agent has dice - if not, skip to next player
            if env.game_state.player_dice_count[current_learning_agent] == 0:
                # Learning agent has no dice, skip to next player
                env.game_state.next_player()
                self.current_players[i] = env.game_state.current_player
                self.active_agent_ids[i] = self.current_players[i]
                # Update current learning agent
                if all_learn:
                    current_learning_agent = self.active_agent_ids[i]
                # Check if game ended after skipping
                if env.game_state.game_over:
                    done = True
                    self.episode_already_done[i] = True
                    # Get observation for learning agent (even though they have no dice)
                    obs = env.get_observation_for_player(current_learning_agent)
                    observations_list.append(obs)
                    rewards.append(0.0)
                    dones.append(True)
                    
                    # Get episode statistics from environment (before reset)
                    bid_count = getattr(env, 'episode_bid_count', 0)
                    challenge_count = getattr(env, 'episode_challenge_count', 0)
                    believe_count = getattr(env, 'episode_believe_count', 0)
                    invalid_action_count = getattr(env, 'episode_invalid_action_count', 0)
                    winner = env.game_state.winner if hasattr(env.game_state, "winner") and env.game_state.winner is not None else -1
                    
                    learning_agent_id = 0
                    
                    # Get reward config
                    reward_config = getattr(env, 'reward_config', None)
                    if reward_config is None:
                        try:
                            from ..training.config import DEFAULT_CONFIG
                            reward_config = DEFAULT_CONFIG.reward
                        except ImportError:
                            reward_config = None
                    
                    # Calculate final reward using accumulated reward
                    accumulated_reward = self.vecmonitor_accumulated_reward[i]
                    final_reward = self._calculate_final_reward(env, learning_agent_id, accumulated_reward, reward_config)
                    
                    # Get episode length
                    if all_learn:
                        episode_length = self.all_agents_episode_length[i].get(current_learning_agent, 0)
                    else:
                        episode_length = self.learning_agent_episode_length[i]
                    
                    # Print episode summary
                    learning_agent_won = winner == 0
                    win_status = "WIN" if learning_agent_won else "DEFEAT"
                    dice_str = str(list(env.game_state.player_dice_count))
                    print(f"{win_status} | reward: {final_reward:.2f} | stats: bids={bid_count}, challenges={challenge_count}, believe={believe_count}, invalid={invalid_action_count} | dice: {dice_str}")
                    
                    # Calculate step_reward for VecMonitor (difference between final and accumulated)
                    step_reward = final_reward - accumulated_reward
                    
                    # Create episode info dict
                    info = {
                        "game_over": True,
                        "winner": winner,
                        "episode": {
                            "r": float(final_reward),
                            "l": int(episode_length),
                            "bid_count": bid_count,
                            "challenge_count": challenge_count,
                            "believe_count": believe_count,
                            "invalid_action_count": invalid_action_count,
                            "winner": winner,
                        },
                        "episode_reward": float(final_reward),
                        "episode_length": int(episode_length),
                    }
                    
                    observations_list.append(obs)
                    rewards.append(step_reward)
                    dones.append(True)
                    infos.append(info)
                    # Reset VecMonitor accumulated reward tracker for next episode
                    self.vecmonitor_accumulated_reward[i] = 0.0
                    continue
            
            # Get action: use PPO action for agent 0, or for current agent in all_learn mode
            if current_learning_agent == 0:
                # Use action from PPO (provided by step_async)
                # MaskablePPO automatically uses action_masks() method during training
                action = int(self._actions[i])
            else:
                # In all_learn mode, get action from current model using predict
                # This allows all agents to learn, but we still collect experience
                obs_for_agent = env.get_observation_for_player(current_learning_agent)
                if isinstance(obs_for_agent, dict):
                    obs_for_predict = {key: np.array([value]) for key, value in obs_for_agent.items()}
                    
                    # Extract action mask if available and pass it to predict()
                    action_masks = None
                    if "action_mask" in obs_for_agent:
                        mask = obs_for_agent["action_mask"]
                        if isinstance(mask, np.ndarray):
                            mask = mask.astype(bool)
                            if mask.ndim > 1:
                                mask = mask.flatten()
                            action_masks = np.array([mask])
                        elif isinstance(mask, (list, tuple)):
                            mask = np.array(mask, dtype=bool)
                            if mask.ndim > 1:
                                mask = mask.flatten()
                            action_masks = np.array([mask])
                    
                    # Predict with action mask if available
                    if action_masks is not None:
                        action, _ = self.current_model.predict(obs_for_predict, deterministic=False, action_masks=action_masks)
                    else:
                        action, _ = self.current_model.predict(obs_for_predict, deterministic=False)
                else:
                    obs_for_predict = obs_for_agent.reshape(1, -1)
                    action, _ = self.current_model.predict(obs_for_predict, deterministic=False)
                action = int(action[0])
            
            env.set_active_player(current_learning_agent)

            # Execute action (invalid actions now give -1 reward and pass turn, no retry)
            obs, reward, terminated, truncated_flag, info = env.step(action)
            done = terminated or truncated_flag

            # Get action validity from info to only count valid actions in episode_length
            action_valid = info.get("action_info", {}).get("action_valid", True)
            
            # Accumulate reward for learning agent during episode
            # In normal mode: only agent 0 accumulates reward
            # In all_learn mode: only agent 0 accumulates reward (for VecMonitor compatibility)
            if current_learning_agent == 0:
                # Accumulate reward in vecmonitor_accumulated_reward (single source of truth)
                self.vecmonitor_accumulated_reward[i] += reward
                
                # Track episode length for statistics
                if action_valid:
                    if all_learn:
                        if current_learning_agent not in self.all_agents_episode_length[i]:
                            self.all_agents_episode_length[i][current_learning_agent] = 0
                        self.all_agents_episode_length[i][current_learning_agent] += 1
                    else:
                        self.learning_agent_episode_length[i] += 1
                
                if done:
                    self.episode_already_done[i] = True
            elif done:
                # Episode ended on opponent's turn
                self.episode_already_done[i] = True

            # Update current player
            self.current_players[i] = env.game_state.current_player
            self.active_agent_ids[i] = self.current_players[i]

            # ===== ШАГ 2: ТЕПЕРЬ ОППОНЕНТЫ ХОДЯТ, ПОКА СНОВА НЕ НАСТАНЕТ ХОД LEARNING AGENT =====
            # In all_learn mode, we don't need to advance to next learning agent
            # because all agents are learning and we collect experience from all of them
            if not all_learn:
                opponent_step_count = 0
                max_opponent_steps = 100  # Защита от бесконечного цикла
                actual_num_players = env.num_players  # Get actual number of players for this episode

                while not done and self.active_agent_ids[i] != 0 and opponent_step_count < max_opponent_steps:
                    opponent_step_count += 1

                    # CRITICAL: Skip players with no dice - they have already lost and cannot make moves
                    # Players with 0 dice are eliminated and should be automatically skipped
                    # Use next_player() which automatically skips players with 0 dice
                    if env.game_state.player_dice_count[self.active_agent_ids[i]] == 0:
                        env.game_state.next_player()
                        self.current_players[i] = env.game_state.current_player
                        self.active_agent_ids[i] = self.current_players[i]
                        # Check if game ended after skipping players
                        if env.game_state.game_over:
                            done = True
                            self.episode_already_done[i] = True
                        # Continue to next iteration to check next player
                        continue

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
                            
                            # Extract action mask if available and pass it to predict()
                            action_masks = None
                            if "action_mask" in obs_for_opp:
                                mask = obs_for_opp["action_mask"]
                                if isinstance(mask, np.ndarray):
                                    mask = mask.astype(bool)
                                    if mask.ndim > 1:
                                        mask = mask.flatten()
                                    action_masks = np.array([mask])
                                elif isinstance(mask, (list, tuple)):
                                    mask = np.array(mask, dtype=bool)
                                    if mask.ndim > 1:
                                        mask = mask.flatten()
                                    action_masks = np.array([mask])
                            
                            # Predict with action mask if available
                            if action_masks is not None:
                                action, _ = opponent_model.predict(obs_for_predict, deterministic=False, action_masks=action_masks)
                            else:
                                action, _ = opponent_model.predict(obs_for_predict, deterministic=False)
                        else:
                            obs_for_predict = obs_for_opp.reshape(1, -1)
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
                        # If episode ended on opponent's turn, we'll override episode info with learning agent's stats later
                        if done:
                            # Episode ended on opponent's turn - copy non-episode fields, episode info will be set later
                            if isinstance(opp_info, dict):
                                info = {k: v for k, v in opp_info.items() if k not in ("episode", "episode_reward", "episode_length", "winner", "game_over")}
                            else:
                                info = {}
                        else:
                            # Episode not done, just use opponent's info as-is
                            info = opp_info
                    else:
                        # No opponent model, skip this step
                        # Use actual number of players for this episode
                        self.active_agent_ids[i] = (self.active_agent_ids[i] + 1) % actual_num_players
                        break

            # ===== ШАГ 3: ПРОВЕРКА ЗАВЕРШЕНИЯ И ОБНОВЛЕНИЕ СТАТИСТИКИ =====
            if done and self.opponent_pool is not None:
                winner = env.game_state.winner if hasattr(env.game_state, "winner") and env.game_state.winner is not None else -1
                won = winner == 0  # Learning agent is agent 0
                # Update winrate statistics for opponents
                actual_num_players = env.num_players
                for opp_idx in range(1, actual_num_players):
                    if opp_idx - 1 < len(self.opponent_paths[i]):
                        opponent_path = self.opponent_paths[i][opp_idx - 1]
                        if opponent_path is not None:
                            self.opponent_pool.update_winrate(opponent_path, won)

            # Process episode info when done=True
            # We MUST use learning agent's statistics, not opponent's or environment's
            # This is because episode can end on opponent's turn, but we need stats for learning agent
            if done:
                # Get episode statistics from environment
                bid_count = getattr(env, 'episode_bid_count', 0)
                challenge_count = getattr(env, 'episode_challenge_count', 0)
                believe_count = getattr(env, 'episode_believe_count', 0)
                invalid_action_count = getattr(env, 'episode_invalid_action_count', 0)
                winner = env.game_state.winner if hasattr(env.game_state, "winner") and env.game_state.winner is not None else -1
                
                learning_agent_id = 0
                
                # Get reward config
                reward_config = getattr(env, 'reward_config', None)
                if reward_config is None:
                    try:
                        from ..training.config import DEFAULT_CONFIG
                        reward_config = DEFAULT_CONFIG.reward
                    except ImportError:
                        reward_config = None
                
                # Calculate final reward using accumulated reward
                accumulated_reward = self.vecmonitor_accumulated_reward[i]
                final_reward = self._calculate_final_reward(env, learning_agent_id, accumulated_reward, reward_config)
                
                # Get episode length
                if all_learn:
                    episode_length = self.all_agents_episode_length[i].get(learning_agent_id, 0)
                else:
                    episode_length = self.learning_agent_episode_length[i]
                
                # Print episode summary
                learning_agent_won = winner == 0
                win_status = "WIN" if learning_agent_won else "DEFEAT"
                dice_str = str(list(env.game_state.player_dice_count))
                print(f"{win_status} | reward: {final_reward:.2f} | stats: bids={bid_count}, challenges={challenge_count}, believe={believe_count}, invalid={invalid_action_count} | dice: {dice_str}")
                
                # Ensure info dict exists
                if not isinstance(info, dict):
                    info = {}
                
                # Create episode info with learning agent's statistics (overwrite any existing episode info)
                info["episode"] = {
                    "r": float(final_reward),
                    "l": int(episode_length),
                    "bid_count": bid_count,
                    "challenge_count": challenge_count,
                    "believe_count": believe_count,
                    "invalid_action_count": invalid_action_count,
                    "winner": winner,
                }
                info["episode_reward"] = float(final_reward)
                info["episode_length"] = int(episode_length)
                info["winner"] = winner
                info["game_over"] = True

            # Get observation for learning agent
            if obs is None:
                if all_learn:
                    obs = env.get_observation_for_player(current_learning_agent)
                else:
                    obs = env.get_observation_for_player(0)
            elif not all_learn and self.active_agent_ids[i] != 0:
                obs = env.get_observation_for_player(0)

            observations_list.append(obs)
            
            # Calculate step_reward for VecMonitor
            # VecMonitor accumulates rewards from step_reward on every step
            if done:
                # Episode ended: calculate step_reward as difference between final reward and accumulated reward
                if "episode" in info and "r" in info["episode"]:
                    final_episode_reward = info["episode"]["r"]
                    step_reward = final_episode_reward - self.vecmonitor_accumulated_reward[i]
                else:
                    # No episode info (shouldn't happen), return 0.0
                    step_reward = 0.0
                # Reset accumulated reward for next episode
                self.vecmonitor_accumulated_reward[i] = 0.0
            else:
                # During episode: return reward for learning agent (agent 0), 0.0 for opponent
                if current_learning_agent == 0:
                    # Learning agent's step, return actual reward (already accumulated in vecmonitor_accumulated_reward above)
                    step_reward = reward
                else:
                    # Opponent's step, learning agent gets 0 reward
                    step_reward = 0.0
            
            rewards.append(step_reward)
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

        # Store observations for action_masks() method
        self.last_obs = observations

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
        # Special handling for action_masks: return masks from VecEnv, not from individual envs
        if method_name == "action_masks":
            # Return action masks from VecEnv (this method is defined on VecEnv level)
            masks = self.action_masks()
            if indices is None:
                indices = list(range(self.num_envs))
            # Return masks for requested indices
            return [masks[i] for i in indices]
        
        if indices is None:
            indices = list(range(self.num_envs))
        return [
            getattr(self.envs[i], method_name)(*method_args, **method_kwargs)
            for i in indices
        ]

    def action_masks(self) -> np.ndarray:
        """
        Get action masks for all environments.
        
        This method is required by MaskablePPO to support action masking.
        It gets action_mask directly from each environment for the learning agent (agent 0).
        
        CRITICAL: This method is called by MaskablePPO before predict() to get current action masks.
        It must return masks for the learning agent (agent 0) based on the CURRENT game state.
        
        Returns:
            Array of action masks with shape (num_envs, action_space.n)
        """
        masks = []
        for i, env in enumerate(self.envs):
            # CRITICAL: Always get mask for learning agent (agent 0) in normal mode
            # In all_learn mode, use current active agent
            # But for MaskablePPO, we always want masks for agent 0 (the learning agent)
            # because only agent 0 makes moves through VecEnv in normal mode
            if self.all_agents_learn_mode[i]:
                # In all_learn mode, use current active agent
                agent_id = self.active_agent_ids[i]
            else:
                # Normal mode: always use agent 0 (learning agent)
                agent_id = 0
            
            # IMPORTANT: Get observation for the agent who will make the move
            # This should be the learning agent (agent 0) in normal mode
            # Make sure we get the observation for the correct agent and current game state
            try:
                obs = env.get_observation_for_player(agent_id)
                
                # Extract action_mask from observation
                if isinstance(obs, dict):
                    if "action_mask" in obs:
                        action_mask = obs["action_mask"].copy()  # Make a copy to avoid issues
                        # Ensure it's boolean and 1D array
                        if action_mask.dtype != bool:
                            action_mask = action_mask.astype(bool)
                        # Flatten if needed (should be 1D already, but ensure)
                        if action_mask.ndim > 1:
                            action_mask = action_mask.flatten()
                        # Ensure correct size
                        if len(action_mask) != self.action_space.n:
                            # Size mismatch, return all actions as valid (fallback)
                            print(f"Warning: Action mask size mismatch for env {i}: "
                                  f"expected {self.action_space.n}, got {len(action_mask)}")
                            masks.append(np.ones(self.action_space.n, dtype=bool))
                        else:
                            masks.append(action_mask)
                    else:
                        # No action_mask in observation, return all actions as valid
                        masks.append(np.ones(self.action_space.n, dtype=bool))
                else:
                    # Array observations don't have action_mask, return all actions as valid
                    masks.append(np.ones(self.action_space.n, dtype=bool))
            except Exception as e:
                # If there's an error getting observation, return all actions as valid (fallback)
                print(f"Warning: Error getting action mask for env {i}, agent {agent_id}: {e}")
                masks.append(np.ones(self.action_space.n, dtype=bool))
        
        # Convert list of masks to array with shape (num_envs, action_space.n)
        if len(masks) == 0:
            return np.ones((self.num_envs, self.action_space.n), dtype=bool)
        
        # Ensure all masks have the same shape
        masks_array = np.array(masks)
        if masks_array.shape != (self.num_envs, self.action_space.n):
            print(f"Warning: Action masks shape mismatch: expected {(self.num_envs, self.action_space.n)}, "
                  f"got {masks_array.shape}")
            # Return all actions as valid if shape is wrong
            return np.ones((self.num_envs, self.action_space.n), dtype=bool)
        
        return masks_array

