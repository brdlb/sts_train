"""
Opponent manager for handling RL agents and bots.

This module manages creation, sampling, and action retrieval for opponents.
"""

import random
import numpy as np
from typing import List, Optional, Dict, Union, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from sb3_contrib import MaskablePPO
    PPO = MaskablePPO
else:
    PPO = Any

try:
    from ..training.opponent_pool import OpponentPool
except ImportError:
    OpponentPool = None


class OpponentManager:
    """
    Manager for opponents (RL agents and bots).
    
    Responsibilities:
    - Create and load opponents
    - Sample opponents from pool
    - Get actions from opponents
    - Update opponent statistics
    
    Contract:
    - Supports both RL agents and bots
    - Guarantees uniqueness of opponents in one room
    - Caches loaded models for performance
    """
    
    def __init__(
        self,
        use_bot_opponents: bool = True,
        bot_personalities: Optional[List[str]] = None,
        bot_personality_tracker: Optional[Any] = None,
        opponent_pool: Optional[Any] = None,
        current_model: Optional[PPO] = None,
    ):
        """
        Initialize opponent manager.
        
        Args:
            use_bot_opponents: If True, use bot opponents instead of RL agents
            bot_personalities: List of bot personality names to use
            bot_personality_tracker: Tracker for bot personality statistics
            opponent_pool: Pool for sampling RL opponent snapshots
            current_model: Current learning model (for self-play)
        """
        self.use_bot_opponents = use_bot_opponents
        self.bot_personalities = bot_personalities
        self.bot_personality_tracker = bot_personality_tracker
        self.opponent_pool = opponent_pool
        self.current_model = current_model
        
        # Storage for opponents (per environment)
        # Format: List[List[Optional[PPO]]] for RL models
        # Format: List[List[Optional[BotAgent]]] for bots
        self.opponent_models: List[List[Optional[PPO]]] = []
        self.opponent_bots: List[List[Optional[Any]]] = []
        self.opponent_paths: List[List[Optional[str]]] = []
    
    def initialize_for_envs(self, num_envs: int, max_opponents: int) -> None:
        """
        Initialize storage for multiple environments.
        
        Args:
            num_envs: Number of environments
            max_opponents: Maximum number of opponents per environment
        """
        self.opponent_models = [[None] * max_opponents for _ in range(num_envs)]
        self.opponent_bots = [[None] * max_opponents for _ in range(num_envs)]
        self.opponent_paths = [[None] * max_opponents for _ in range(num_envs)]
    
    def create_opponents_for_env(
        self,
        env_idx: int,
        env,
        num_players: int,
        current_step: int = 0,
        opponent_snapshot_ids: Optional[List[Optional[str]]] = None,
    ) -> bool:
        """
        Create opponents for a specific environment.
        
        Args:
            env_idx: Environment index
            env: Environment instance
            num_players: Actual number of players for this episode
            current_step: Current training step
            opponent_snapshot_ids: Optional list of snapshot IDs to use
            
        Returns:
            True if all agents should learn (pool empty), False otherwise
        """
        num_opponents = num_players - 1
        
        if self.use_bot_opponents:
            self._create_bot_opponents(env_idx, env, num_players)
            return False  # Bots don't learn
        elif self.opponent_pool is not None:
            return self._sample_rl_opponents(
                env_idx, env, num_players, current_step, opponent_snapshot_ids
            )
        else:
            # No pool, all agents use current model (self-play)
            for slot_idx in range(num_opponents):
                self.opponent_models[env_idx][slot_idx] = self.current_model
                self.opponent_paths[env_idx][slot_idx] = None
            return True  # All agents learn
    
    def _create_bot_opponents(
        self,
        env_idx: int,
        env,
        num_players: int,
    ) -> None:
        """Create bot opponents for environment."""
        try:
            from ..agents.bot_agent import BotAgent
            from ..agents.bot_logic.constants import BOT_PERSONALITIES
            from ..agents.bot_logic.player_analysis import create_initial_player_analysis
        except ImportError:
            print("Warning: Bot agents not available, falling back to self-play")
            return
        
        num_opponents = num_players - 1
        
        # Get available personalities
        if self.bot_personalities is None:
            available_personalities = list(BOT_PERSONALITIES.values())
        else:
            available_personalities = [
                BOT_PERSONALITIES[name]
                for name in self.bot_personalities
                if name in BOT_PERSONALITIES
            ]
            if not available_personalities:
                available_personalities = list(BOT_PERSONALITIES.values())
        
        # Create player analysis for all players
        player_analysis = {}
        for player_id in range(num_players):
            player_analysis[player_id] = create_initial_player_analysis(player_id)
        
        # Shuffle personalities to ensure random selection without duplicates
        personalities_pool = available_personalities.copy()
        random.shuffle(personalities_pool)
        
        # Extend pool if needed
        while len(personalities_pool) < num_opponents:
            additional_personalities = available_personalities.copy()
            random.shuffle(additional_personalities)
            personalities_pool.extend(additional_personalities)
        
        # Create bots for each opponent slot
        for slot_idx in range(num_opponents):
            personality = personalities_pool[slot_idx]
            opponent_id = slot_idx + 1  # Agent 0 is learning agent
            bot = BotAgent(
                agent_id=opponent_id,
                personality=personality,
                env=env,
                player_analysis=player_analysis,
            )
            self.opponent_bots[env_idx][slot_idx] = bot
            self.opponent_models[env_idx][slot_idx] = None
            self.opponent_paths[env_idx][slot_idx] = None
    
    def _sample_rl_opponents(
        self,
        env_idx: int,
        env,
        num_players: int,
        current_step: int,
        opponent_snapshot_ids: Optional[List[Optional[str]]] = None,
    ) -> bool:
        """Sample RL opponents from pool."""
        num_opponents = num_players - 1
        
        # Check if pool has snapshots
        if not self.opponent_pool.snapshots:
            # Pool is empty, all opponents use current model (self-play)
            for slot_idx in range(num_opponents):
                self.opponent_models[env_idx][slot_idx] = self.current_model
                self.opponent_paths[env_idx][slot_idx] = None
            return True  # All agents learn
        
        # Get available snapshot IDs
        available_snapshot_ids = list(self.opponent_pool.snapshots.keys())
        
        # If specific snapshot IDs provided, use them (ensure uniqueness)
        if opponent_snapshot_ids is not None:
            unique_snapshot_ids = []
            seen_ids = set()
            for snap_id in opponent_snapshot_ids:
                if snap_id is not None and snap_id not in seen_ids and snap_id in available_snapshot_ids:
                    unique_snapshot_ids.append(snap_id)
                    seen_ids.add(snap_id)
        else:
            # Sample unique snapshots from pool
            shuffled_ids = available_snapshot_ids.copy()
            random.shuffle(shuffled_ids)
            unique_snapshot_ids = shuffled_ids[:num_opponents]
        
        # Assign unique snapshots to opponent slots
        num_unique_snapshots = len(unique_snapshot_ids)
        for slot_idx, snapshot_id in enumerate(unique_snapshot_ids):
            if slot_idx >= num_opponents:
                break
            
            opponent_path = self.opponent_pool.get_snapshot_by_id(snapshot_id)
            if opponent_path and env is not None:
                try:
                    opponent_model = self.opponent_pool.load_snapshot(opponent_path, env)
                    if opponent_model is not None:
                        self.opponent_models[env_idx][slot_idx] = opponent_model
                        self.opponent_paths[env_idx][slot_idx] = opponent_path
                        if snapshot_id in self.opponent_pool.snapshots:
                            self.opponent_pool.snapshots[snapshot_id].last_used = current_step
                    else:
                        self.opponent_models[env_idx][slot_idx] = self.current_model
                        self.opponent_paths[env_idx][slot_idx] = None
                except Exception as e:
                    print(f"Warning: Failed to load opponent snapshot {opponent_path}: {e}")
                    self.opponent_models[env_idx][slot_idx] = self.current_model
                    self.opponent_paths[env_idx][slot_idx] = None
            else:
                self.opponent_models[env_idx][slot_idx] = self.current_model
                self.opponent_paths[env_idx][slot_idx] = None
        
        # Fill remaining slots with current model
        for slot_idx in range(num_unique_snapshots, num_opponents):
            self.opponent_models[env_idx][slot_idx] = self.current_model
            self.opponent_paths[env_idx][slot_idx] = None
        
        # Check if all slots use current model
        all_use_current_model = all(
            self.opponent_models[env_idx][i] == self.current_model
            for i in range(num_opponents)
        )
        return all_use_current_model
    
    def get_opponent_action(
        self,
        env_idx: int,
        opponent_idx: int,
        observation: Union[Dict, np.ndarray],
        deterministic: bool = False,
    ) -> int:
        """
        Get action from opponent.
        
        Args:
            env_idx: Environment index
            opponent_idx: Opponent index (0-based)
            observation: Observation for opponent
            deterministic: Whether to use deterministic policy
            
        Returns:
            Action from opponent
        """
        if self.use_bot_opponents:
            bot = self.opponent_bots[env_idx][opponent_idx]
            if bot is not None:
                return bot.act(observation, deterministic=deterministic)
            return 0  # Fallback: challenge
        
        # Use RL agent
        opponent_model = self.opponent_models[env_idx][opponent_idx]
        if opponent_model is not None:
            if isinstance(observation, dict):
                obs_for_predict = {key: np.array([value]) for key, value in observation.items()}
                
                # Extract action mask if available
                action_masks = None
                if "action_mask" in observation:
                    mask = observation["action_mask"]
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
                
                if action_masks is not None:
                    action, _ = opponent_model.predict(
                        obs_for_predict, 
                        deterministic=deterministic, 
                        action_masks=action_masks
                    )
                else:
                    action, _ = opponent_model.predict(obs_for_predict, deterministic=deterministic)
            else:
                obs_for_predict = observation.reshape(1, -1)
                action, _ = opponent_model.predict(obs_for_predict, deterministic=deterministic)
            return int(action[0])
        
        return 0  # Fallback: challenge
    
    def update_opponent_statistics(
        self,
        env_idx: int,
        winner: int,
        num_players: int,
    ) -> None:
        """
        Update opponent statistics after episode.
        
        Args:
            env_idx: Environment index
            winner: Winner ID (0 = learning agent)
            num_players: Number of players in game
        """
        # Update RL opponent statistics
        if self.opponent_pool is not None:
            won = winner == 0  # Learning agent is agent 0
            for opp_idx in range(1, num_players):
                if opp_idx - 1 < len(self.opponent_paths[env_idx]):
                    opponent_path = self.opponent_paths[env_idx][opp_idx - 1]
                    if opponent_path is not None:
                        self.opponent_pool.update_winrate(opponent_path, won)
        
        # Update bot statistics if using bots
        if self.use_bot_opponents and self.bot_personality_tracker is not None:
            # This is handled separately in perudo_vec_env.py via _update_bot_statistics
            # because it requires access to env instance for detailed statistics
            pass

