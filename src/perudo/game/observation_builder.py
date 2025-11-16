"""
Observation builder for agents.

This module handles building observations and action masks for agents.
"""

import numpy as np
from typing import Dict
from .interfaces import IObservationBuilder
from ..training.config import ObservationConfig, DEFAULT_CONFIG
from ..utils.helpers import create_observation_dict, create_action_mask
from .rules import PerudoRules


class ObservationBuilder(IObservationBuilder):
    """
    Builder for agent observations.
    
    Responsibilities:
    - Create observations in Dict[str, np.ndarray] format
    - Generate action masks
    - Validate observations (optionally)
    
    Contract:
    - Observations always match observation_space
    - Action masks always have correct size
    - Thread-safe
    """
    
    def __init__(
        self, 
        config: ObservationConfig = None,
        max_quantity: int = 30,
        action_space_size: int = None
    ):
        """
        Initialize observation builder.
        
        Args:
            config: Observation configuration (uses DEFAULT_CONFIG if not provided)
            max_quantity: Maximum dice quantity for bids
            action_space_size: Size of action space (calculated if not provided)
        """
        self.config = config if config is not None else DEFAULT_CONFIG.environment.observation
        self.max_quantity = max_quantity
        
        # Calculate action space size if not provided
        if action_space_size is None:
            from ..utils.helpers import get_action_space_size
            self.action_space_size = get_action_space_size(
                max_players=self.config.max_players,
                max_quantity=self.max_quantity
            )
        else:
            self.action_space_size = action_space_size
    
    def build_observation(
        self, 
        game_state, 
        player_id: int
    ) -> Dict[str, np.ndarray]:
        """
        Build observation for player.
        
        Args:
            game_state: Current game state
            player_id: ID of player
            
        Returns:
            Observation in Dict[str, np.ndarray] format
        """
        player_dice = game_state.get_player_dice(player_id)
        
        observation = create_observation_dict(
            current_bid=game_state.current_bid,
            bid_history=game_state.bid_history,
            player_dice_count=game_state.player_dice_count,
            current_player=game_state.current_player,
            palifico_active=game_state.palifico_active,
            believe_called=game_state.believe_called,
            player_dice=player_dice,
            max_history_length=self.config.max_history_length,
            max_players=self.config.max_players,
            agent_id=player_id,
            num_agents=self.config.max_players,
        )
        
        # Add action mask if configured
        if self.config.include_action_mask:
            action_mask = self.get_action_mask(game_state, player_id)
            observation["action_mask"] = action_mask
        
        # Validate if configured
        if self.config.validate_observations:
            self._validate_observation(observation)
        
        return observation
    
    def get_action_mask(
        self, 
        game_state, 
        player_id: int
    ) -> np.ndarray:
        """
        Get mask of available actions.
        
        Args:
            game_state: Current game state
            player_id: ID of player
            
        Returns:
            Boolean array of size action_space.n
        """
        available_actions = PerudoRules.get_available_actions(game_state, player_id)
        action_mask = create_action_mask(
            available_actions, 
            self.action_space_size, 
            self.max_quantity
        )
        return action_mask
    
    def _validate_observation(self, observation: Dict[str, np.ndarray]) -> None:
        """
        Validate observation structure and types.
        
        Args:
            observation: Observation dictionary
            
        Raises:
            ValueError: If observation is invalid
        """
        required_keys = ["bid_history", "static_info"]
        if self.config.include_action_mask:
            required_keys.append("action_mask")
        
        for key in required_keys:
            if key not in observation:
                raise ValueError(f"Missing required key in observation: {key}")
            
            if not isinstance(observation[key], np.ndarray):
                raise ValueError(f"Observation key '{key}' must be numpy array")
        
        # Validate bid_history shape
        expected_history_shape = (self.config.max_history_length, 3)
        if observation["bid_history"].shape != expected_history_shape:
            raise ValueError(
                f"bid_history shape mismatch: expected {expected_history_shape}, "
                f"got {observation['bid_history'].shape}"
            )
        
        # Validate action_mask size if present
        if "action_mask" in observation:
            if observation["action_mask"].size != self.action_space_size:
                raise ValueError(
                    f"action_mask size mismatch: expected {self.action_space_size}, "
                    f"got {observation['action_mask'].size}"
                )

