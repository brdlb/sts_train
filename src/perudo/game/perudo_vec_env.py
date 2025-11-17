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
    from ..training.rule_based_pool import RuleBasedOpponentPool
    from ..agents.rule_based_agent import RuleBasedAgent
    from ..agents.bot_personalities import BOT_PERSONALITIES
except ImportError:
    RuleBasedOpponentPool = None  # type: ignore
    RuleBasedAgent = None  # type: ignore
    BOT_PERSONALITIES = None  # type: ignore

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
    
    # Maximum steps to prevent infinite loops when resetting or advancing to learning agent
    MAX_RESET_STEPS = 100

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
        rule_based_pool: Optional[Any] = None,  # RuleBasedOpponentPool type
        training_mode: str = "selfplay",  # "selfplay", "botplay", "mixed"
        mixed_mode_ratio: float = 0.5,  # Ratio of botplay in mixed mode
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
            rule_based_pool: Rule-based opponent pool for botplay mode
            training_mode: Training mode ("selfplay", "botplay", "mixed")
            mixed_mode_ratio: Ratio of botplay in mixed mode (0.0 to 1.0)
        """
        self.num_envs = num_envs
        self.random_num_players = random_num_players
        self.min_players = min_players
        self.max_players = max_players
        self.max_quantity = max_quantity
        self.max_history_length = max_history_length if max_history_length is not None else history_length
        
        # Use maximum number of players for observation space
        if random_num_players:
            self.max_num_players = max(max_players, num_players)  # At least max_players for random selection
        else:
            self.max_num_players = num_players
        
        self.num_players = num_players  # Will be updated per environment in reset() if random_num_players=True
        self.opponent_pool = opponent_pool
        self.current_model = current_model
        self.rule_based_pool = rule_based_pool
        self.training_mode = training_mode
        self.mixed_mode_ratio = mixed_mode_ratio

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
                # debug_moves is not a parameter of PerudoEnv, it's handled by VecEnv
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
        
        # Rule-based agents for each environment
        # Initialize with maximum size (7 opponents for 8 players max, agent 0 is learning agent)
        self.opponent_agents: List[List[Optional[RuleBasedAgent]]] = [
            [None] * (self.max_num_players - 1) for _ in range(num_envs)
        ]
        
        # Track personality keys for each rule-based bot (for statistics)
        # Format: List[List[Optional[str]]] - personality_key for each opponent slot
        self.opponent_personality_keys: List[List[Optional[str]]] = [
            [None] * (self.max_num_players - 1) for _ in range(num_envs)
        ]
        
        # Track steps for each rule-based bot in current game
        # Format: List[List[int]] - step count for each opponent slot
        self.bot_steps: List[List[int]] = [
            [0] * (self.max_num_players - 1) for _ in range(num_envs)
        ]

        # Track which agent is active in each environment
        # CRITICAL: This is synchronized with env.game_state.current_player
        # Always use _sync_active_agent_id() to update this value
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
        
        # Debug mode flag (will be set from training script)
        self.debug_mode = None

        # Get observation and action spaces from first environment
        obs_space = self.envs[0].observation_space
        action_space = self.envs[0].action_space

        super().__init__(num_envs, obs_space, action_space)

    def _sync_active_agent_id(self, env_idx: int) -> None:
        """
        Synchronize active_agent_ids[env_idx] with env.game_state.current_player.
        
        This is the single source of truth for current player tracking.
        Always use this method to update active_agent_ids.
        
        Args:
            env_idx: Environment index
        """
        env = self.envs[env_idx]
        self.active_agent_ids[env_idx] = env.game_state.current_player
    
    def _validate_active_agent_id(self, env_idx: int) -> bool:
        """
        Validate that active_agent_ids[env_idx] matches game_state.current_player.
        
        Args:
            env_idx: Environment index
            
        Returns:
            True if synchronized, False otherwise
        """
        env = self.envs[env_idx]
        expected = env.game_state.current_player
        actual = self.active_agent_ids[env_idx]
        if actual != expected:
            print(f"Warning: active_agent_ids[{env_idx}]={actual} != game_state.current_player={expected}. "
                  f"Auto-syncing...", flush=True)
            self._sync_active_agent_id(env_idx)
            return False
        return True

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
            # Call for both RL opponent pool and rule-based pool (botplay mode)
            if self.opponent_pool is not None or self.rule_based_pool is not None:
                self._sample_opponents_for_env(
                    i, current_step, actual_num_players, opponent_snapshot_ids
                )
            
            # Synchronize active_agent_ids with game_state.current_player (single source of truth)
            self._sync_active_agent_id(i)
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

            # Advance to learning agent's turn if needed
            # Skip only if not in all_learn mode (in all_learn mode, all agents are learning)
            if not self.all_agents_learn_mode[i]:
                steps = 0
                actual_num_players = env.num_players  # Get actual number of players for this episode
                while self.active_agent_ids[i] != 0 and steps < self.MAX_RESET_STEPS:
                    steps += 1
                    
                    # Skip players with no dice
                    if self._skip_players_without_dice(i):
                        # Game ended after skipping players, reset again
                        obs, info = env.reset(seed=seeds[i], options=options[i])
                        # Re-sample opponents after reset
                        if self.opponent_pool is not None or self.rule_based_pool is not None:
                            self._sample_opponents_for_env(i, current_step, env.num_players, opponent_snapshot_ids)
                        # Synchronize active_agent_ids with game_state.current_player
                        self._sync_active_agent_id(i)
                        steps = 0  # Start over
                        continue
                    
                    # Validate synchronization before checking
                    self._validate_active_agent_id(i)
                    
                    # Skip if current player is learning agent
                    if self.active_agent_ids[i] == 0:
                        break
                    
                    # Execute opponent move (will validate opponent_id matches current_player)
                    done, obs_for_opp = self._execute_opponent_move(i, self.active_agent_ids[i], context="reset")
                    
                    if done:
                        # Episode ended before learning agent's turn, reset again
                        # These episodes should not be counted as learning agent made no moves
                        obs, info = env.reset(seed=seeds[i], options=options[i])
                        # Re-sample opponents after reset
                        if self.opponent_pool is not None or self.rule_based_pool is not None:
                            self._sample_opponents_for_env(i, current_step, env.num_players, opponent_snapshot_ids)
                        # Synchronize active_agent_ids with game_state.current_player
                        self._sync_active_agent_id(i)
                        steps = 0  # Start over
                        continue

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
        # Use actual number of players if provided, otherwise use max
        if num_players is None:
            num_players = self.max_num_players

        # Number of opponent slots (agents 1, 2, ..., num_players-1)
        num_opponents = num_players - 1
        
        # Clear rule-based agents for this environment
        for opp_idx in range(num_opponents):
            self.opponent_agents[env_idx][opp_idx] = None
        
        # Determine which pool to use based on training_mode
        use_rule_based = False
        if self.training_mode == "botplay":
            use_rule_based = True
        elif self.training_mode == "mixed":
            # Randomly choose based on mixed_mode_ratio
            use_rule_based = random.random() < self.mixed_mode_ratio
        
        # Debug: verify training mode and rule_based_pool availability
        if self.training_mode == "botplay" and self.rule_based_pool is None:
            print(f"Warning: botplay mode but rule_based_pool is None for env {env_idx}", flush=True)
        
        # Handle botplay mode
        if use_rule_based:
            if self.rule_based_pool is None:
                # Fallback to selfplay if rule_based_pool is not available
                use_rule_based = False
            else:
                # Sample rule-based bots with uniqueness constraint
                selected_personalities = []  # Track selected personalities to ensure uniqueness
                bot_names = []  # Store bot names for printing
                for opp_idx in range(num_opponents):
                    personality_name = self.rule_based_pool.sample(exclude_personalities=selected_personalities)
                    if BOT_PERSONALITIES and personality_name in BOT_PERSONALITIES:
                        personality = BOT_PERSONALITIES[personality_name]
                        agent = RuleBasedAgent(
                            agent_id=opp_idx + 1,  # Opponents are 1-indexed
                            personality=personality,
                            max_quantity=self.max_quantity,
                            max_players=self.max_num_players,
                            max_history_length=self.max_history_length,
                        )
                        self.opponent_agents[env_idx][opp_idx] = agent
                        # Store personality key for statistics tracking
                        self.opponent_personality_keys[env_idx][opp_idx] = personality_name
                        # Reset step counter for this bot
                        self.bot_steps[env_idx][opp_idx] = 0
                        # Add to selected list to ensure uniqueness
                        selected_personalities.append(personality_name)
                        # Store bot name for printing (use personality.name if available)
                        if hasattr(personality, 'name') and personality.name:
                            bot_names.append(personality.name)
                        else:
                            # Fallback to personality_name if name is not available
                            bot_names.append(personality_name)
                        # Clear PPO model for this slot
                        self.opponent_models[env_idx][opp_idx] = None
                        self.opponent_paths[env_idx][opp_idx] = None
                
                # Print bot names at the start of each game in botplay mode
                # Disabled - player names output turned off
                # if self.training_mode == "botplay" and bot_names:
                #     env_prefix = f"[Env {env_idx}] " if self.num_envs > 1 else ""
                #     # Use flush=True to ensure output is immediately visible
                #     print(f"{env_prefix}🎮 Новая игра! Игроки: {', '.join(bot_names)}", flush=True)
                
                self.all_agents_learn_mode[env_idx] = False
                return
        
        # Handle selfplay mode (original logic)
        if self.opponent_pool is None:
            # If no opponent pool, all opponents use current model (self-play)
            if self.current_model is None:
                # Model not yet created, opponents will be set later when model is available
                # For now, leave them as None - they will be updated when current_model is set
                pass
            else:
                for opp_idx in range(1, num_players):
                    self.opponent_models[env_idx][opp_idx - 1] = self.current_model
                    self.opponent_paths[env_idx][opp_idx - 1] = None
            self.all_agents_learn_mode[env_idx] = True
            return

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
    
    def update_opponent_models_for_current_model(self):
        """
        Update opponent models to use current_model for all environments.
        This should be called when current_model is set after initialization.
        """
        if self.current_model is None:
            return
        
        # Update opponents for all environments
        for env_idx in range(self.num_envs):
            # Get actual number of players for this environment
            env = self.envs[env_idx]
            num_players = env.num_players if hasattr(env, 'num_players') else self.max_num_players
            num_opponents = num_players - 1
            
            # Only update if opponent_pool is None (self-play mode)
            # or if opponent_pool is empty
            if self.opponent_pool is None:
                for opp_idx in range(1, num_players):
                    if opp_idx - 1 < len(self.opponent_models[env_idx]):
                        self.opponent_models[env_idx][opp_idx - 1] = self.current_model
                        self.opponent_paths[env_idx][opp_idx - 1] = None
            elif not self.opponent_pool.snapshots:
                # Pool is empty, update opponents to use current model
                for opp_idx in range(1, num_players):
                    if opp_idx - 1 < len(self.opponent_models[env_idx]):
                        self.opponent_models[env_idx][opp_idx - 1] = self.current_model
                        self.opponent_paths[env_idx][opp_idx - 1] = None

    def step_async(self, actions: np.ndarray) -> None:
        """
        Step all environments asynchronously.

        Args:
            actions: Array of actions for all environments
        """
        self._actions = actions

    def _skip_players_without_dice(self, env_idx: int) -> bool:
        """
        Skip players with no dice for a specific environment.
        
        Updates active_agent_ids to sync with game_state.current_player.
        Uses game_state.current_player as source of truth, not active_agent_ids.
        
        Args:
            env_idx: Environment index
            
        Returns:
            True if game ended after skipping players, False otherwise
        """
        env = self.envs[env_idx]
        
        # CRITICAL: If game is already over, don't try to skip players
        # This prevents infinite loops when learning agent wins
        # (all other players are eliminated, trying to skip them would loop forever)
        if env.game_state.game_over:
            return True
        
        # CRITICAL: Always check game_state.current_player, not active_agent_ids
        # This ensures we catch cases where they are out of sync
        current_player = env.game_state.current_player
        
        # Skip players with no dice - they have already lost and cannot make moves
        # Use next_player() which automatically skips players with 0 dice
        if (current_player < len(env.game_state.player_dice_count) and 
            env.game_state.player_dice_count[current_player] == 0):
            env.game_state.next_player()
            # Synchronize active_agent_ids with game_state.current_player (single source of truth)
            self._sync_active_agent_id(env_idx)
            
            # Validate synchronization
            self._validate_active_agent_id(env_idx)
            
            # Check if game ended after skipping players
            if env.game_state.game_over:
                return True
        
        return False
    
    def _execute_opponent_move(
        self, 
        env_idx: int, 
        opponent_id: int, 
        context: str = "step"
    ) -> Tuple[bool, Optional[Any]]:
        """
        Execute opponent move for a specific environment.
        
        Handles both PPO models and rule-based agents.
        
        Args:
            env_idx: Environment index
            opponent_id: Opponent agent ID
            context: Context for debugging ("reset" or "step")
            
        Returns:
            Tuple of (done, obs):
            - done: True if episode ended after opponent's move
            - obs: Observation after opponent's move, or None if episode ended
        """
        env = self.envs[env_idx]
        
        # CRITICAL: Validate that opponent_id matches game_state.current_player
        # This ensures we're executing move for the correct player
        if opponent_id != env.game_state.current_player:
            print(f"Warning: opponent_id={opponent_id} != game_state.current_player={env.game_state.current_player} "
                  f"in _execute_opponent_move (env {env_idx}, context={context}). "
                  f"Syncing active_agent_ids and skipping move.", flush=True)
            # Sync and skip to next player
            self._sync_active_agent_id(env_idx)
            env.game_state.next_player()
            self._sync_active_agent_id(env_idx)
            return False, None
        
        opponent_idx = opponent_id - 1
        
        # Check if this opponent is a rule-based agent
        opponent_agent = None
        if opponent_idx < len(self.opponent_agents[env_idx]):
            opponent_agent = self.opponent_agents[env_idx][opponent_idx]
        
        # Get opponent model if not using rule-based agent
        opponent_model = None
        if opponent_agent is None:
            if opponent_idx < len(self.opponent_models[env_idx]):
                opponent_model = self.opponent_models[env_idx][opponent_idx]
        
        # No opponent model or agent available
        if opponent_agent is None and opponent_model is None:
            # Skip to next player
            env.game_state.next_player()
            # Synchronize active_agent_ids with game_state.current_player
            self._sync_active_agent_id(env_idx)
            return False, None
        
        # Get observation for opponent
        obs_for_opp = env.get_observation_for_player(opponent_id)
        
        # Get action from opponent
        if opponent_agent is not None:
            # Rule-based bot
            action = opponent_agent.get_action(obs_for_opp, action_mask=None)
            # Increment step counter for this bot
            if opponent_idx < len(self.bot_steps[env_idx]):
                self.bot_steps[env_idx][opponent_idx] += 1
        elif opponent_model is not None:
            # PPO model
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
        else:
            # Should not happen, but handle gracefully
            env.game_state.next_player()
            # Synchronize active_agent_ids with game_state.current_player
            self._sync_active_agent_id(env_idx)
            return False, None
        
        # Debug mode: log turn transfer to opponent
        if self.debug_mode is not None and self.debug_mode.is_set():
            try:
                prev_player = self.active_agent_ids[env_idx] if env_idx < len(self.active_agent_ids) else "unknown"
                print(f"[DEBUG TURN] Env {env_idx}: Передача хода -> Player {opponent_id} (было: {prev_player}, контекст: {context})")
                # Wait for user input before opponent's move
                try:
                    input("[DEBUG TURN] Нажмите Enter для продолжения...")
                except (EOFError, KeyboardInterrupt):
                    pass
            except Exception:
                pass
        
        # Execute opponent action
        env.set_active_player(opponent_id)
        prev_current_player = env.game_state.current_player
        obs_for_opp, _, terminated, opp_truncated, opp_info = env.step(action)
        done = terminated or opp_truncated
        new_current_player = env.game_state.current_player
        
        # Debug mode: print opponent's move and turn transfer
        if self.debug_mode is not None and self.debug_mode.is_set():
            try:
                from ..utils.helpers import action_to_bid
                action_type, param1, param2 = action_to_bid(action, env.max_quantity)
                
                player_name = f"Player {opponent_id}"
                if action_type == "bid":
                    move_str = f"{player_name}: BID {param1}x{param2}"
                elif action_type == "challenge":
                    move_str = f"{player_name}: CHALLENGE"
                elif action_type == "believe":
                    move_str = f"{player_name}: BELIEVE"
                else:
                    move_str = f"{player_name}: {action_type}"
                
                # Show current game state
                current_bid = env.game_state.current_bid
                bid_str = f"{current_bid[0]}x{current_bid[1]}" if current_bid else "none"
                dice_str = str(list(env.game_state.player_dice_count))
                
                # Show turn transfer
                turn_transfer = f" | Ход: Player {prev_current_player} -> Player {new_current_player}"
                
                print(f"[DEBUG] {move_str} | Current bid: {bid_str} | Dice: {dice_str}{turn_transfer}")
                
                # Wait for user input after opponent's move
                try:
                    input("[DEBUG TURN] Нажмите Enter для продолжения...")
                except (EOFError, KeyboardInterrupt):
                    # Handle case where stdin is not available or user cancels
                    pass
            except Exception:
                # Silently ignore errors to prevent crashes
                pass
        
        # Update current player from game state (env.step already updated current_player)
        # CRITICAL: Always synchronize with game_state.current_player (single source of truth)
        prev_active = self.active_agent_ids[env_idx] if env_idx < len(self.active_agent_ids) else None
        self._sync_active_agent_id(env_idx)
        
        # Validate synchronization
        self._validate_active_agent_id(env_idx)
        
        # Debug mode: log active agent update
        if self.debug_mode is not None and self.debug_mode.is_set() and prev_active is not None:
            try:
                if prev_active != self.active_agent_ids[env_idx]:
                    print(f"[DEBUG TURN] Env {env_idx}: Обновление active_agent_ids: {prev_active} -> {self.active_agent_ids[env_idx]}")
            except Exception:
                pass
        
        if done:
            self.episode_already_done[env_idx] = True
        
        return done, obs_for_opp
    
    def _handle_episode_end(
        self, 
        env_idx: int, 
        env: PerudoEnv,
        learning_agent_id: int = 0
    ) -> Dict[str, Any]:
        """
        Handle episode end and calculate final reward and statistics.
        
        Args:
            env_idx: Environment index
            env: Environment instance
            learning_agent_id: ID of learning agent (usually 0)
            
        Returns:
            Episode info dictionary with reward, length, and statistics
        """
        all_learn = self.all_agents_learn_mode[env_idx]
        
        # Get episode statistics from environment
        bid_count = getattr(env, 'episode_bid_count', 0)
        challenge_count = getattr(env, 'episode_challenge_count', 0)
        believe_count = getattr(env, 'episode_believe_count', 0)
        invalid_action_count = getattr(env, 'episode_invalid_action_count', 0)
        winner = env.game_state.winner if hasattr(env.game_state, "winner") and env.game_state.winner is not None else -1
        
        # Get reward config
        reward_config = getattr(env, 'reward_config', None)
        if reward_config is None:
            try:
                from ..training.config import DEFAULT_CONFIG
                reward_config = DEFAULT_CONFIG.reward
            except ImportError:
                reward_config = None
        
        # Calculate final reward using accumulated reward
        accumulated_reward = self.vecmonitor_accumulated_reward[env_idx]
        final_reward = self._calculate_final_reward(env, learning_agent_id, accumulated_reward, reward_config)
        
        # Get episode length
        if all_learn:
            episode_length = self.all_agents_episode_length[env_idx].get(learning_agent_id, 0)
        else:
            episode_length = self.learning_agent_episode_length[env_idx]
        
        # Calculate bid history usage percentage
        bid_history = env.game_state.bid_history
        # Count valid actions (encoded_bid > 0 indicates valid action, padding has encoded_bid = 0)
        # action_type: 0=bid, 1=challenge, 2=believe
        valid_bids = [action for action in bid_history if action[1] > 0]  # encoded_bid > 0
        history_length = len(valid_bids)
        max_history_length = env.max_history_length
        # Calculate percentage: min(history_length, max_history_length) / max_history_length * 100
        history_usage_percent = min(history_length, max_history_length) / max_history_length * 100.0
        
        # Print episode summary
        learning_agent_won = winner == learning_agent_id
        win_status = "WIN" if learning_agent_won else "DEFEAT"
        dice_str = str(list(env.game_state.player_dice_count))
        print(f"{win_status} | reward: {final_reward:.2f} | stats: bids={bid_count}, challenges={challenge_count}, believe={believe_count}, invalid={invalid_action_count} | dice: {dice_str} | history: {history_length}/{max_history_length} ({history_usage_percent:.1f}%)")
        
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
                "history_length": history_length,
                "history_usage_percent": history_usage_percent,
            },
            "episode_reward": float(final_reward),
            "episode_length": int(episode_length),
            "history_length": history_length,
            "history_usage_percent": history_usage_percent,
        }
        
        return info
    
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

            # Check if episode was already done - automatically reset to start new episode
            # This prevents infinite loops where done=True is returned repeatedly
            if self.episode_already_done[i]:
                try:
                    # Reset environment (this will randomly select number of players 3-8)
                    obs, info = env.reset()
                    
                    # Get actual number of players for this episode
                    actual_num_players = env.num_players
                    
                    # Reset step counters for all bots
                    for opp_idx in range(actual_num_players - 1):
                        if opp_idx < len(self.bot_steps[i]):
                            self.bot_steps[i][opp_idx] = 0
                    
                    # Sample opponents from pool for this environment
                    if self.opponent_pool is not None or self.rule_based_pool is not None:
                        self._sample_opponents_for_env(i, self.current_step, actual_num_players)
                    
                    # Synchronize active_agent_ids with game_state.current_player (single source of truth)
                    self._sync_active_agent_id(i)
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
                    all_learn = self.all_agents_learn_mode[i]
                    if not all_learn:
                        steps = 0
                        actual_num_players = env.num_players
                        while self.active_agent_ids[i] != 0 and steps < self.MAX_RESET_STEPS:
                            steps += 1
                            
                            # Skip players with no dice
                            if self._skip_players_without_dice(i):
                                # Game ended after skipping players, reset again
                                obs, info = env.reset()
                                # Reset step counters for all bots
                                for opp_idx in range(env.num_players - 1):
                                    if opp_idx < len(self.bot_steps[i]):
                                        self.bot_steps[i][opp_idx] = 0
                                # Re-sample opponents after reset
                                if self.opponent_pool is not None or self.rule_based_pool is not None:
                                    self._sample_opponents_for_env(i, self.current_step, env.num_players)
                                # Synchronize active_agent_ids with game_state.current_player
                                self._sync_active_agent_id(i)
                                # Reset episode statistics
                                self.learning_agent_episode_reward[i] = 0.0
                                self.learning_agent_episode_length[i] = 0
                                self.all_agents_episode_reward[i] = {}
                                self.all_agents_episode_length[i] = {}
                                self.vecmonitor_accumulated_reward[i] = 0.0
                                steps = 0  # Start over
                                continue
                            
                            # Validate synchronization before checking
                            self._validate_active_agent_id(i)
                            
                            # Skip if current player is learning agent
                            if self.active_agent_ids[i] == 0:
                                break
                            
                            # Execute opponent move (will validate opponent_id matches current_player)
                            done, obs_for_opp = self._execute_opponent_move(i, self.active_agent_ids[i], context="step")
                            
                            if done:
                                # If game ended before learning agent's turn, reset again
                                obs, info = env.reset()
                                # Reset step counters for all bots
                                for opp_idx in range(env.num_players - 1):
                                    if opp_idx < len(self.bot_steps[i]):
                                        self.bot_steps[i][opp_idx] = 0
                                # Re-sample opponents after reset
                                if self.opponent_pool is not None or self.rule_based_pool is not None:
                                    self._sample_opponents_for_env(i, self.current_step, env.num_players)
                                # Synchronize active_agent_ids with game_state.current_player
                                self._sync_active_agent_id(i)
                                # Reset episode statistics
                                self.learning_agent_episode_reward[i] = 0.0
                                self.learning_agent_episode_length[i] = 0
                                self.all_agents_episode_reward[i] = {}
                                self.all_agents_episode_length[i] = {}
                                self.vecmonitor_accumulated_reward[i] = 0.0
                                steps = 0  # Start over
                                continue
                except (ValueError, IndexError, KeyError) as e:
                    # Critical errors that should be propagated
                    raise
                except Exception as e:
                    # Other errors: log and continue to prevent missing results
                    import traceback
                    print(f"Warning: Exception in episode_already_done reset loop for env {i}: {e}")
                    print(traceback.format_exc())
                
                # Get observation for learning agent
                # Always use agent 0 (learning agent) regardless of all_learn mode for VecEnv compatibility
                obs = env.get_observation_for_player(0)
                done = False
                reward = 0.0
                info = {}
                
                observations_list.append(obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
                continue

            # Step 1: Learning agent makes a move
            # CRITICAL: Always use agent 0 as learning agent for VecEnv compatibility
            # Only agent 0 makes moves through VecEnv, regardless of all_learn mode
            all_learn = self.all_agents_learn_mode[i]
            learning_agent_id = 0
            
            # Debug mode: log turn transfer to learning agent
            if self.debug_mode is not None and self.debug_mode.is_set():
                try:
                    prev_active = self.active_agent_ids[i] if i < len(self.active_agent_ids) else "unknown"
                    current_player_from_state = env.game_state.current_player
                    print(f"[DEBUG TURN] Env {i}: Ход обученного агента (Player {learning_agent_id}) | "
                          f"active_agent_ids={prev_active}, game_state.current_player={current_player_from_state}")
                    # Wait for user input before learning agent's move
                    try:
                        input("[DEBUG TURN] Нажмите Enter для продолжения...")
                    except (EOFError, KeyboardInterrupt):
                        pass
                except Exception:
                    pass
            
            # Check if learning agent has dice - if not, episode should end (learning agent lost)
            if env.game_state.player_dice_count[learning_agent_id] == 0:
                # CRITICAL: If learning agent has no dice, episode must end immediately
                # This prevents infinite loops where opponents play forever while learning agent is eliminated
                # The learning agent has lost (no dice remaining), so episode ends even if game continues for others
                done = True
                self.episode_already_done[i] = True
                
                # Get observation for learning agent (even though they have no dice)
                obs = env.get_observation_for_player(learning_agent_id)
                
                # Handle episode end - learning agent lost (no dice)
                info = self._handle_episode_end(i, env, learning_agent_id)
                
                # Calculate step_reward for VecMonitor (difference between final and accumulated)
                step_reward = info["episode"]["r"] - self.vecmonitor_accumulated_reward[i]
                
                observations_list.append(obs)
                rewards.append(step_reward)
                dones.append(True)
                infos.append(info)
                # Reset VecMonitor accumulated reward tracker for next episode
                self.vecmonitor_accumulated_reward[i] = 0.0
                continue
            
            # Get action: always use PPO action for agent 0 (learning agent)
            # MaskablePPO automatically uses action_masks() method during training
            action = int(self._actions[i])
            
            env.set_active_player(learning_agent_id)

            # Execute action (invalid actions now give -1 reward and pass turn, no retry)
            prev_player_before_action = env.game_state.current_player
            obs, reward, terminated, truncated_flag, info = env.step(action)
            done = terminated or truncated_flag
            new_player_after_action = env.game_state.current_player
            
            # Debug mode: log turn transfer after learning agent's action
            if self.debug_mode is not None and self.debug_mode.is_set() and prev_player_before_action != new_player_after_action:
                try:
                    print(f"[DEBUG TURN] Env {i}: После хода обученного агента: "
                          f"Player {prev_player_before_action} -> Player {new_player_after_action}")
                except Exception:
                    pass

            # Get action validity from info to only count valid actions in episode_length
            action_valid = info.get("action_info", {}).get("action_valid", True)
            
            # Debug mode: print every move
            if self.debug_mode is not None and self.debug_mode.is_set():
                try:
                    from ..utils.helpers import action_to_bid
                    action_type, param1, param2 = action_to_bid(action, env.max_quantity)
                    
                    player_name = f"Player {learning_agent_id}" if all_learn else "Learning Agent"
                    if action_type == "bid":
                        move_str = f"{player_name}: BID {param1}x{param2}"
                    elif action_type == "challenge":
                        move_str = f"{player_name}: CHALLENGE"
                    elif action_type == "believe":
                        move_str = f"{player_name}: BELIEVE"
                    else:
                        move_str = f"{player_name}: {action_type}"
                    
                    # Show current game state
                    current_bid = env.game_state.current_bid
                    bid_str = f"{current_bid[0]}x{current_bid[1]}" if current_bid else "none"
                    dice_str = str(list(env.game_state.player_dice_count))
                    
                    # Show turn transfer after learning agent's action
                    turn_transfer = f" | Ход: Player {prev_player_before_action} -> Player {new_player_after_action}"
                    
                    print(f"[DEBUG] {move_str} | Current bid: {bid_str} | Dice: {dice_str} | Reward: {reward:.2f}{turn_transfer}")
                    
                    # Wait for user input after learning agent's move
                    try:
                        input("[DEBUG TURN] Нажмите Enter для продолжения...")
                    except (EOFError, KeyboardInterrupt):
                        # Handle case where stdin is not available or user cancels
                        pass
                except Exception:
                    # Silently ignore errors to prevent crashes
                    pass
            
            # Accumulate reward for learning agent during episode
            # Always accumulate reward for agent 0 (learning agent)
            # VecMonitor expects rewards for the agent making moves through VecEnv (agent 0)
            # Accumulate reward in vecmonitor_accumulated_reward (single source of truth)
            self.vecmonitor_accumulated_reward[i] += reward
            
            # Track episode length for statistics
            if action_valid:
                if all_learn:
                    if learning_agent_id not in self.all_agents_episode_length[i]:
                        self.all_agents_episode_length[i][learning_agent_id] = 0
                    self.all_agents_episode_length[i][learning_agent_id] += 1
                else:
                    self.learning_agent_episode_length[i] += 1
            
            # CRITICAL: If game ended after learning agent's move, handle episode end immediately
            # This ensures game completes correctly when learning agent wins
            if done:
                self.episode_already_done[i] = True
                
                # CRITICAL: Do NOT update current_player or active_agent_ids when game is over
                # This prevents infinite loops when trying to skip players without dice
                # When game ends, current_player may point to a player without dice,
                # and updating active_agent_ids would trigger _skip_players_without_dice
                # which would call next_player() in an infinite loop
                # Keep current player as is - episode is ending anyway
                
                # Skip opponent loop - game is already over
                # Continue to episode end handling below
            else:
                # Game not over, update current player and continue with opponent turns
                # CRITICAL: Always synchronize with game_state.current_player (single source of truth)
                prev_active = self.active_agent_ids[i] if i < len(self.active_agent_ids) else None
                self._sync_active_agent_id(i)
                
                # Validate synchronization
                self._validate_active_agent_id(i)
                
                # Debug mode: log active agent update after learning agent's action
                if self.debug_mode is not None and self.debug_mode.is_set() and prev_active is not None:
                    try:
                        if prev_active != self.active_agent_ids[i]:
                            print(f"[DEBUG TURN] Env {i}: Обновление после хода обученного агента: "
                                  f"active_agent_ids {prev_active} -> {self.active_agent_ids[i]}, "
                                  f"game_state.current_player={env.game_state.current_player}")
                    except Exception:
                        pass

                # Step 2: Opponents take turns until learning agent's turn again
                # CRITICAL: Even in all_learn mode, we need to execute opponent moves
                # because VecEnv only handles actions for agent 0, and other agents
                # (even if they use the same model) need to make moves through _execute_opponent_move
                # The difference is that in all_learn mode, all opponents use the current model,
                # but we still need to execute their moves to advance the game state
                opponent_step_count = 0
                actual_num_players = env.num_players

                # Debug mode: log entry into opponent loop
                if self.debug_mode is not None and self.debug_mode.is_set():
                    try:
                        print(f"[DEBUG TURN] Env {i}: Вход в цикл противников | "
                              f"current_player={env.game_state.current_player}, "
                              f"active_agent_ids={self.active_agent_ids[i]}, "
                              f"done={done}, all_learn={all_learn}")
                    except Exception:
                        pass

                while not done and opponent_step_count < self.MAX_RESET_STEPS:
                    opponent_step_count += 1
                    
                    # CRITICAL: Always read from game_state.current_player (single source of truth)
                    # Validate synchronization before each iteration
                    self._validate_active_agent_id(i)
                    current_player = env.game_state.current_player
                    
                    # Skip if current player is learning agent
                    if current_player == 0:
                        # Debug mode: log that we reached learning agent's turn
                        if self.debug_mode is not None and self.debug_mode.is_set():
                            try:
                                print(f"[DEBUG TURN] Env {i}: Достигнут ход обученного агента (Player 0), выход из цикла противников")
                            except Exception:
                                pass
                        break

                    # Skip players with no dice
                    skip_result = self._skip_players_without_dice(i)
                    if skip_result:
                        # Game ended after skipping players
                        done = True
                        self.episode_already_done[i] = True
                        # Continue to next iteration to check next player
                        continue
                    
                    # Validate synchronization after skipping
                    self._validate_active_agent_id(i)
                    current_player = env.game_state.current_player
                    
                    # Skip if current player is learning agent (may have changed after skipping)
                    if current_player == 0:
                        break

                    # Debug mode: log starting opponent move
                    if self.debug_mode is not None and self.debug_mode.is_set():
                        try:
                            print(f"[DEBUG TURN] Env {i}: Начало хода противника Player {current_player} "
                                  f"(шаг противника {opponent_step_count}/{self.MAX_RESET_STEPS})")
                        except Exception:
                            pass

                    # Execute opponent move (will validate opponent_id matches current_player)
                    done, obs_for_opp = self._execute_opponent_move(i, current_player, context="step")

                    # Store info for later use
                    # If episode ended on opponent's turn, we'll override episode info with learning agent's stats later
                    if done:
                        # Episode ended on opponent's turn - copy non-episode fields, episode info will be set later
                        info = {}
                        # Break out of opponent loop if game ended
                        break
                    else:
                        # Episode not done, continue to next opponent if not learning agent's turn
                        # _execute_opponent_move already updated current_player to next player
                        # Continue loop to check if next player is learning agent (player 0)
                        info = {}
                        # active_agent_ids[i] already updated in _execute_opponent_move from env.game_state.current_player
                        # Loop will continue and check if active_agent_ids[i] == 0

            # Step 3: Check episode end and update statistics
            if done:
                winner = env.game_state.winner if hasattr(env.game_state, "winner") and env.game_state.winner is not None else -1
                actual_num_players = env.num_players
                
                # Update statistics for RL opponents
                if self.opponent_pool is not None:
                    won = winner == learning_agent_id  # Learning agent is agent 0
                    # Update winrate statistics for opponents
                    for opp_idx in range(1, actual_num_players):
                        if opp_idx - 1 < len(self.opponent_paths[i]):
                            opponent_path = self.opponent_paths[i][opp_idx - 1]
                            if opponent_path is not None:
                                self.opponent_pool.update_winrate(opponent_path, won)
                
                # Update statistics for rule-based bots
                if self.rule_based_pool is not None:
                    # Collect all participants (both bots and RL agent if present)
                    participants = []
                    
                    # Add RL agent if present (player_id 0)
                    if learning_agent_id == 0:
                        participants.append((None, 0, 0))  # None for personality_key, 0 for player_id
                    
                    # Add all rule-based bots
                    for opp_idx in range(1, actual_num_players):
                        if opp_idx - 1 < len(self.opponent_agents[i]):
                            agent = self.opponent_agents[i][opp_idx - 1]
                            if agent is not None:
                                personality_key = self.opponent_personality_keys[i][opp_idx - 1]
                                steps = self.bot_steps[i][opp_idx - 1] if opp_idx - 1 < len(self.bot_steps[i]) else 0
                                won_bot = (winner == (opp_idx + 1))  # Opponents are 1-indexed in game, but stored as 0-indexed
                                
                                # Update game result for this bot
                                if personality_key is not None:
                                    self.rule_based_pool.update_game_result(
                                        personality_key=personality_key,
                                        won=won_bot,
                                        steps=steps,
                                    )
                                    participants.append((personality_key, opp_idx + 1, steps))
                    
                    # Update ELO for all participating rule-based bots
                    if len(participants) > 0:
                        self.rule_based_pool.update_elos_for_game(
                            participants=participants,
                            winner_id=winner,
                        )

            # Process episode info when done=True
            # We MUST use learning agent's statistics, not opponent's or environment's
            # This is because episode can end on opponent's turn, but we need stats for learning agent
            if done:
                # Handle episode end
                episode_info = self._handle_episode_end(i, env, learning_agent_id)
                
                # Ensure info dict exists
                if not isinstance(info, dict):
                    info = {}
                
                # Update info with episode statistics
                info.update(episode_info)
                
                # Debug mode: wait for user input before continuing
                if self.debug_mode is not None and self.debug_mode.is_set():
                    try:
                        print("\n[DEBUG MODE] Game ended. Press Enter to continue...")
                        input()
                    except (EOFError, KeyboardInterrupt):
                        # Handle case where stdin is not available or user cancels
                        pass
                    except Exception:
                        # Silently ignore errors to prevent crashes
                        pass

            # Get observation for learning agent
            # Always use agent 0 (learning agent) regardless of all_learn mode for VecEnv compatibility
            if obs is None or (not all_learn and self.active_agent_ids[i] != 0):
                obs = env.get_observation_for_player(learning_agent_id)

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
                # During episode: always return reward for learning agent (agent 0)
                # Learning agent's step, return actual reward (already accumulated in vecmonitor_accumulated_reward above)
                step_reward = reward
            
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
            # CRITICAL: Always return masks for agent 0 (learning agent) regardless of all_learn mode
            # MaskablePPO expects masks for the agent making moves through VecEnv (agent 0)
            # Only agent 0 makes moves through VecEnv, so we always use agent 0 for action masks
            agent_id = 0
            
            # IMPORTANT: Get observation for the learning agent (agent 0)
            # This ensures action masks match the agent that will make the move through VecEnv
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

