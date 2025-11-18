"""
Opponent management module for Perudo vectorized environment.

Handles opponent sampling, bot creation, action retrieval, and statistics tracking.
"""

import random
from typing import List, Optional, Dict, Any, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sb3_contrib import MaskablePPO
    PPO = MaskablePPO
else:
    PPO = Any


class OpponentManager:
    """
    Manages opponents (RL agents or bots) for each environment.
    
    Handles:
    - Sampling RL opponents from pool
    - Creating bot opponents
    - Getting actions from opponents
    - Updating opponent statistics
    """
    
    def __init__(
        self,
        num_envs: int,
        max_num_players: int,
        opponent_pool: Optional[Any] = None,  # OpponentPool type
        current_model: Optional[PPO] = None,
        use_bot_opponents: bool = True,
        bot_personalities: Optional[List[str]] = None,
        bot_personality_tracker: Optional[Any] = None,
    ):
        """
        Initialize opponent manager.
        
        Args:
            num_envs: Number of parallel environments
            max_num_players: Maximum number of players per environment
            opponent_pool: Opponent pool for sampling RL opponents
            current_model: Current learning model (for self-play)
            use_bot_opponents: If True, use rule-based bots instead of RL agents
            bot_personalities: List of bot personality names to use
            bot_personality_tracker: Bot personality tracker for statistics
        """
        self.num_envs = num_envs
        self.max_num_players = max_num_players
        self.opponent_pool = opponent_pool
        self.current_model = current_model
        self.use_bot_opponents = use_bot_opponents
        self.bot_personalities = bot_personalities
        self.bot_personality_tracker = bot_personality_tracker
        
        # Opponent models for each environment (RL agents)
        # Initialize with maximum size (opponents for max_num_players, agent 0 is learning agent)
        self.opponent_models: List[List[Optional[PPO]]] = [
            [None] * (max_num_players - 1) for _ in range(num_envs)
        ]
        
        # Opponent paths for each environment (to track which snapshot was used)
        self.opponent_paths: List[List[Optional[str]]] = [
            [None] * (max_num_players - 1) for _ in range(num_envs)
        ]
        
        # Bot opponents for each environment
        self.opponent_bots: List[List[Optional[Any]]] = [  # BotAgent type
            [None] * (max_num_players - 1) for _ in range(num_envs)
        ]
        
        # Track if all agents should learn (when opponent pool is empty or using self-play)
        self.all_agents_learn_mode: List[bool] = [False] * num_envs
    
    def sample_opponents(
        self,
        env_idx: int,
        env: Any,
        current_step: int = 0,
        num_players: Optional[int] = None,
        opponent_snapshot_ids: Optional[List[Optional[str]]] = None,
    ) -> None:
        """
        Sample opponents from pool for a specific environment.
        
        If snapshots exist in the pool, only unique snapshots are added to the room.
        Remaining opponent slots use the current model (self-play), just like in the first iteration.
        
        Args:
            env_idx: Environment index
            env: Environment instance
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
        
        # Note: This function should only be called when NOT using bot opponents
        # Bot opponents are created directly via create_bot_opponents
        if self.opponent_pool is None:
            # If no opponent pool, all agents learn
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
            if opponent_path and env is not None:
                # Load opponent model
                try:
                    opponent_model = self.opponent_pool.load_snapshot(
                        opponent_path, env
                    )
                    if opponent_model is not None:
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
    
    def create_bot_opponents(
        self,
        env_idx: int,
        env: Any,
        num_players: int,
    ) -> None:
        """
        Create bot opponents for a specific environment.
        
        Args:
            env_idx: Environment index
            env: Environment instance
            num_players: Actual number of players for this episode
        """
        try:
            from ..agents.bot_agent import BotAgent
            from ..agents.bot_logic.constants import BOT_PERSONALITIES
            from ..agents.bot_logic.player_analysis import create_initial_player_analysis
        except ImportError:
            print("Warning: Bot agents not available, falling back to self-play")
            self.all_agents_learn_mode[env_idx] = True
            return
        
        num_opponents = num_players - 1
        
        # Get available personalities
        if self.bot_personalities is None:
            # Use all personalities
            available_personalities = list(BOT_PERSONALITIES.values())
        else:
            # Use only specified personalities
            available_personalities = [
                BOT_PERSONALITIES[name]
                for name in self.bot_personalities
                if name in BOT_PERSONALITIES
            ]
            if not available_personalities:
                # Fallback to all if none specified
                available_personalities = list(BOT_PERSONALITIES.values())
        
        # Create player analysis for all players
        player_analysis = {}
        for player_id in range(num_players):
            player_analysis[player_id] = create_initial_player_analysis(player_id)
        
        # Shuffle personalities to ensure random selection without duplicates
        # Create a copy to avoid modifying the original list
        personalities_pool = available_personalities.copy()
        random.shuffle(personalities_pool)
        
        # If we need more opponents than available personalities, extend the pool
        # by adding shuffled copies until we have enough
        while len(personalities_pool) < num_opponents:
            additional_personalities = available_personalities.copy()
            random.shuffle(additional_personalities)
            personalities_pool.extend(additional_personalities)
        
        # Create bots for each opponent slot
        for slot_idx in range(num_opponents):
            # Select personality from shuffled pool (no duplicates within one room)
            personality = personalities_pool[slot_idx]
            
            # Create bot agent (opponent_id = slot_idx + 1, since agent 0 is learning agent)
            opponent_id = slot_idx + 1
            bot = BotAgent(
                agent_id=opponent_id,
                personality=personality,
                env=env,
                player_analysis=player_analysis,
            )
            self.opponent_bots[env_idx][slot_idx] = bot
            self.opponent_models[env_idx][slot_idx] = None  # Clear RL model
            self.opponent_paths[env_idx][slot_idx] = None
        
        # Bots don't learn, so all_agents_learn_mode is False
        self.all_agents_learn_mode[env_idx] = False
    
    def get_opponent_action(
        self,
        env_idx: int,
        opponent_idx: int,
        player_id: int,
        observation: Union[Dict, np.ndarray],
    ) -> int:
        """
        Get action from opponent (either RL agent or bot).
        
        Args:
            env_idx: Environment index
            opponent_idx: Opponent index (0-based, for opponent slot)
            player_id: Player ID in game
            observation: Observation for the opponent
            
        Returns:
            Action from opponent
        """
        # Check if using bot opponents
        if self.use_bot_opponents:
            bot = self.opponent_bots[env_idx][opponent_idx]
            if bot is not None:
                return bot.act(observation, deterministic=False)
            else:
                # Fallback: challenge
                return 0
        
        # Use RL agent
        opponent_model = self.opponent_models[env_idx][opponent_idx]
        if opponent_model is not None:
            # Handle both dict and array observations
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
                
                # Predict with action mask if available
                if action_masks is not None:
                    action, _ = opponent_model.predict(obs_for_predict, deterministic=False, action_masks=action_masks)
                else:
                    action, _ = opponent_model.predict(obs_for_predict, deterministic=False)
            else:
                obs_for_predict = observation.reshape(1, -1)
                action, _ = opponent_model.predict(obs_for_predict, deterministic=False)
            return int(action[0])
        
        # Fallback: challenge
        return 0
    
    def update_winrates(
        self,
        env_idx: int,
        winner: int,
        won: bool,
        num_players: int,
    ) -> None:
        """
        Update winrate statistics for RL opponents.
        
        Args:
            env_idx: Environment index
            winner: Winner player ID
            won: Whether learning agent won (True if winner == 0)
            num_players: Actual number of players for this episode
        """
        if self.opponent_pool is None:
            return
        
        # Update winrate statistics for opponents
        for opp_idx in range(1, num_players):
            if opp_idx - 1 < len(self.opponent_paths[env_idx]):
                opponent_path = self.opponent_paths[env_idx][opp_idx - 1]
                if opponent_path is not None:
                    self.opponent_pool.update_winrate(opponent_path, won)
    
    def update_bot_statistics(
        self,
        env_idx: int,
        winner: int,
        env: Any,
    ) -> None:
        """
        Update bot personality statistics after a game.
        
        Args:
            env_idx: Environment index
            winner: Winner player ID (0 = learning agent)
            env: Environment instance
        """
        if not self.use_bot_opponents or self.bot_personality_tracker is None:
            return
        
        # Iterate through all bot opponents in this environment
        for slot_idx, bot in enumerate(self.opponent_bots[env_idx]):
            if bot is None:
                continue
            
            bot_player_id = slot_idx + 1  # Bot player IDs start from 1
            personality_name = bot.personality.name
            
            # Check if bot won
            bot_won = (winner == bot_player_id)
            
            # Calculate rounds survived (could use game history or bid history length)
            rounds_survived = len(env.game_state.bid_history)
            
            # Calculate dice lost/won (simplified - could track more accurately)
            initial_dice = env.dice_per_player
            final_dice = env.game_state.player_dice_count[bot_player_id]
            dice_lost = max(0, initial_dice - final_dice)
            dice_won = max(0, final_dice - initial_dice)
            
            # TODO: Track challenges and calzas per bot
            # For now, use simplified statistics
            successful_challenges = 0
            failed_challenges = 0
            successful_calzas = 0
            failed_calzas = 0
            times_challenged = 0
            times_challenged_correctly = 0
            
            # Update statistics
            self.bot_personality_tracker.update_game_result(
                personality_name=personality_name,
                won=bot_won,
                rounds_survived=rounds_survived,
                dice_lost=dice_lost,
                dice_won=dice_won,
                successful_challenges=successful_challenges,
                failed_challenges=failed_challenges,
                successful_calzas=successful_calzas,
                failed_calzas=failed_calzas,
                times_challenged=times_challenged,
                times_challenged_correctly=times_challenged_correctly,
                update_elo=True,
                opponent_elo=None,  # Will use rl_agent_elo from tracker
            )

