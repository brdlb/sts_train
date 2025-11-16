"""
Base abstract class for Perudo environments.

This module provides common functionality for both single and multi-agent environments.
"""

from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Any

from .game_state import GameState
from .game_controller import GameController
from .reward_calculator import RewardCalculator
from .observation_builder import ObservationBuilder
from .episode_tracker import EpisodeTracker
from .debug_logger import DebugLogger
from ..training.config import (
    RewardConfig, 
    EnvironmentConfig, 
    DEFAULT_CONFIG
)


class BasePerudoEnv(gym.Env, ABC):
    """
    Abstract base class for Perudo environments.
    
    Provides common functionality:
    - Game state management
    - Component initialization (controller, reward calculator, etc.)
    - Observation building
    - Common utility methods
    
    Subclasses must implement:
    - step() method
    - reset() method (can call super().reset() for common logic)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        num_players: int = 4,
        dice_per_player: int = 5,
        total_dice_values: int = 6,
        max_quantity: int = 30,
        history_length: int = 10,
        max_history_length: Optional[int] = None,
        render_mode: Optional[str] = None,
        random_num_players: bool = True,
        min_players: int = 3,
        max_players: int = 8,
        reward_config: Optional[RewardConfig] = None,
        environment_config: Optional[EnvironmentConfig] = None,
    ):
        """
        Initialize base Perudo environment.
        
        Args:
            num_players: Number of players
            dice_per_player: Number of dice per player
            total_dice_values: Total possible dice values (usually 6)
            max_quantity: Maximum dice quantity in bid
            history_length: Bid history length (deprecated, use max_history_length)
            max_history_length: Maximum length of bid history sequence
            render_mode: Render mode
            random_num_players: If True, randomly select num_players in each episode
            min_players: Minimum number of players
            max_players: Maximum number of players
            reward_config: Reward configuration (uses DEFAULT_CONFIG.reward if not provided)
            environment_config: Full environment configuration (uses DEFAULT_CONFIG.environment if not provided)
        """
        super().__init__()
        
        # Use environment config if provided, otherwise use defaults
        if environment_config is not None:
            self.environment_config = environment_config
            self.reward_config = environment_config.reward
        else:
            self.environment_config = DEFAULT_CONFIG.environment
            self.reward_config = reward_config if reward_config is not None else DEFAULT_CONFIG.reward
        
        # Store parameters for random player selection
        self.random_num_players = random_num_players
        self.min_players = min_players
        self.max_players = max_players
        
        # Use maximum number of players for observation space
        if random_num_players:
            self.max_num_players = max(max_players, num_players)
        else:
            self.max_num_players = num_players
        
        self.num_players = num_players  # Will be updated in reset() if random_num_players=True
        self.dice_per_player = dice_per_player
        self.total_dice_values = total_dice_values
        self.max_quantity = max_quantity
        self.history_length = history_length
        self.max_history_length = max_history_length if max_history_length is not None else history_length
        self.render_mode = render_mode
        
        # Ensure observation config uses correct history length
        self.environment_config.observation.max_history_length = self.max_history_length
        
        # Initialize components
        self.game_controller = GameController(self.environment_config.game_controller)
        self.reward_calculator = RewardCalculator(self.reward_config)
        self.observation_builder = ObservationBuilder(
            config=self.environment_config.observation,
            max_quantity=max_quantity,
        )
        self.episode_tracker = EpisodeTracker()
        self.debug_logger = DebugLogger(self.environment_config.debug)
        
        # Create game state with initial num_players (will be recreated in reset())
        self.game_state = GameState(
            num_players=num_players,
            dice_per_player=dice_per_player,
            total_dice_values=total_dice_values,
        )
        
        # Define observation space
        static_info_size = (
            self.max_num_players  # agent_id one-hot
            + 2  # current_bid (quantity, value)
            + self.max_num_players  # dice_count
            + 1  # current_player
            + self.max_num_players  # palifico
            + 1  # believe
            + 5  # player_dice
        )
        
        action_size = 2 + max_quantity * 6
        self.observation_space = spaces.Dict({
            "bid_history": spaces.Box(
                low=0, high=100, shape=(self.max_history_length, 3), dtype=np.int32
            ),
            "static_info": spaces.Box(
                low=0, high=100, shape=(static_info_size,), dtype=np.float32
            ),
            "action_mask": spaces.Box(low=0, high=1, shape=(action_size,), dtype=np.bool_),
        })
        
        # Define action space
        self.action_space = spaces.Discrete(action_size)
        
        # Add max_players attribute to observation_space for use by feature extractor
        self.observation_space.max_players = self.max_num_players
        
        # Current active player (for whom observation is returned)
        self.active_player_id = 0
        
        # Information about last action (for debugging)
        self.last_action_info = {}
        
        # Track round information for intermediate rewards
        self.agent_dice_at_round_start = dice_per_player
        self.agent_bid_this_round = False
        self.agent_deferred_reward = 0.0
    
    def get_observation_for_player(self, player_id: int) -> Dict[str, np.ndarray]:
        """
        Get observation for specific player.
        
        Args:
            player_id: Player ID
            
        Returns:
            Observation dictionary
        """
        return self.observation_builder.build_observation(self.game_state, player_id)
    
    def set_active_player(self, player_id: int) -> None:
        """
        Set active player (for whom observation is returned).
        
        Args:
            player_id: Player ID
        """
        self.active_player_id = player_id
    
    def render(self) -> None:
        """Render current game state."""
        if self.render_mode == "human":
            print(f"\n=== Perudo Game State ===")
            print(f"Players: {self.num_players}")
            print(f"Current player: {self.game_state.current_player}")
            print(f"Active player (for observation): {self.active_player_id}")
            if self.game_state.current_bid:
                q, v = self.game_state.current_bid
                print(f"Current bid: {q}x{v}")
            else:
                print("Current bid: none")
            print(f"Player dice counts: {self.game_state.player_dice_count}")
            print(f"Palifico active: {self.game_state.palifico_active}")
            print(f"Believe called: {self.game_state.believe_called}")
            print(f"Game over: {self.game_state.game_over}")
            if self.game_state.winner is not None:
                print(f"Winner: Player {self.game_state.winner}")
            print("=" * 30)
    
    @abstractmethod
    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Optional parameters
            
        Returns:
            Tuple (observation, info)
        """
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute action in environment.
        
        Args:
            action: Action from action_space
            
        Returns:
            Tuple (observation, reward, terminated, truncated, info)
        """
        pass

