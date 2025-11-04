"""
Gymnasium environment for Perudo game.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Any

from .game_state import GameState
from .rules import PerudoRules
from ..utils.helpers import (
    create_observation_vector,
    calculate_reward,
    action_to_bid,
)


class PerudoEnv(gym.Env):
    """Gymnasium environment for Perudo game."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        num_players: int = 4,
        dice_per_player: int = 5,
        total_dice_values: int = 6,
        max_quantity: int = 30,
        history_length: int = 10,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize Perudo environment.

        Args:
            num_players: Number of players
            dice_per_player: Number of dice per player
            total_dice_values: Total possible dice values (usually 6)
            max_quantity: Maximum dice quantity in bid
            history_length: Bid history length in observation
            render_mode: Render mode
        """
        super().__init__()

        self.num_players = num_players
        self.dice_per_player = dice_per_player
        self.total_dice_values = total_dice_values
        self.max_quantity = max_quantity
        self.history_length = history_length
        self.render_mode = render_mode

        # Create game state
        self.game_state = GameState(
            num_players=num_players,
            dice_per_player=dice_per_player,
            total_dice_values=total_dice_values,
        )

        # Define observation space
        # Format: [agent_id(num_players), current_bid(2), history(history_length*3), 
        #          dice_count(num_players), current_player(1), palifico(num_players), 
        #          pacao(1), player_dice(5)]
        obs_size = (
            num_players  # agent_id one-hot
            + 2  # current_bid
            + history_length * 3  # history
            + num_players  # dice_count
            + 1  # current_player
            + num_players  # palifico
            + 1  # pacao
            + 5  # player_dice
        )
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(obs_size,), dtype=np.float32
        )

        # Define action space
        # Actions: 0=challenge, 1=pacao, 2+=bid(encoded)
        action_size = 2 + max_quantity * 6  # 2 special + all bids
        self.action_space = spaces.Discrete(action_size)

        # Current active player (for whom observation is returned)
        self.active_player_id = 0

        # Information about last action (for debugging)
        self.last_action_info = {}

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Optional parameters

        Returns:
            Tuple (observation, info)
        """
        super().reset(seed=seed)

        # Reset game state
        self.game_state.reset()

        # Set active player
        self.active_player_id = self.game_state.current_player

        # Get observation for active player
        observation = self._get_observation(self.active_player_id)

        info = {
            "player_id": self.active_player_id,
            "game_state": self.game_state.get_public_info(),
        }

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action in environment.

        Args:
            action: Action from action_space

        Returns:
            Tuple (observation, reward, terminated, truncated, info)
        """
        # Check that game is not over
        if self.game_state.game_over:
            observation = self._get_observation(self.active_player_id)
            return observation, 0.0, True, False, {"game_over": True}

        # Convert action to game format
        action_type, param1, param2 = action_to_bid(action, self.max_quantity)

        # Check that it's active player's turn
        if self.game_state.current_player != self.active_player_id:
            # If it's not agent's turn, move to next player
            observation = self._get_observation(self.active_player_id)
            return observation, 0.0, False, False, {"error": "Not player's turn"}

        # Execute action
        reward = 0.0
        dice_lost = 0
        challenge_success = None
        pacao_success = None
        action_valid = False

        if action_type == "bid":
            # Make a bid
            quantity, value = param1, param2
            is_valid, error_msg = PerudoRules.is_valid_bid(
                self.game_state, self.active_player_id, quantity, value
            )
            if is_valid:
                action_valid = self.game_state.set_bid(
                    self.active_player_id, quantity, value
                )
                if action_valid:
                    self.game_state.next_player()
                    reward = calculate_reward(
                        "bid", False, -1, self.active_player_id, dice_lost=0
                    )
            else:
                # Invalid action - penalty
                reward = -1.0

        elif action_type == "challenge":
            # Challenge previous player
            can_challenge, error_msg = PerudoRules.can_challenge(
                self.game_state, self.active_player_id
            )
            if can_challenge:
                challenge_success, actual_count, bid_quantity = (
                    self.game_state.challenge_bid(self.active_player_id)
                )
                loser_id, dice_lost = PerudoRules.process_challenge_result(
                    self.game_state,
                    self.active_player_id,
                    challenge_success,
                    actual_count,
                    bid_quantity,
                )
                self.game_state.lose_dice(loser_id, dice_lost)
                self.game_state.current_bid = None
                self.game_state.pacao_called = False

                # Restart round from new player
                if loser_id == self.active_player_id:
                    self.game_state.next_player()
                else:
                    self.game_state.current_player = (loser_id + 1) % self.num_players
                    self.game_state.next_player()

                # Roll dice again
                self.game_state.roll_dice()

                reward = calculate_reward(
                    "challenge",
                    self.game_state.game_over,
                    self.game_state.winner or -1,
                    self.active_player_id,
                    challenge_success=challenge_success,
                    dice_lost=dice_lost if loser_id == self.active_player_id else 0,
                )
                action_valid = True
            else:
                reward = -1.0

        elif action_type == "pacao":
            # Call pacao
            can_pacao, error_msg = PerudoRules.can_call_pacao(
                self.game_state, self.active_player_id
            )
            if can_pacao:
                pacao_success, actual_count = self.game_state.call_pacao(
                    self.active_player_id
                )
                bid_quantity = self.game_state.current_bid[0] if self.game_state.current_bid else 0
                loser_id, dice_lost = PerudoRules.process_pacao_result(
                    self.game_state,
                    self.active_player_id,
                    pacao_success,
                    actual_count,
                    bid_quantity,
                )
                self.game_state.lose_dice(loser_id, dice_lost)
                self.game_state.current_bid = None
                self.game_state.pacao_called = False

                # Restart round
                if loser_id == self.active_player_id:
                    self.game_state.next_player()
                else:
                    self.game_state.current_player = (loser_id + 1) % self.num_players
                    self.game_state.next_player()

                # Roll dice again
                self.game_state.roll_dice()

                reward = calculate_reward(
                    "pacao",
                    self.game_state.game_over,
                    self.game_state.winner or -1,
                    self.active_player_id,
                    pacao_success=pacao_success,
                    dice_lost=dice_lost if loser_id == self.active_player_id else 0,
                )
                action_valid = True
            else:
                reward = -1.0

        # Save action information
        self.last_action_info = {
            "action_type": action_type,
            "action_valid": action_valid,
            "reward": reward,
            "dice_lost": dice_lost,
            "challenge_success": challenge_success,
            "pacao_success": pacao_success,
        }

        # Get new observation
        observation = self._get_observation(self.active_player_id)

        # Check game over
        terminated = self.game_state.game_over
        truncated = False  # Not implemented yet

        info = {
            "player_id": self.active_player_id,
            "game_over": terminated,
            "winner": self.game_state.winner,
            "action_info": self.last_action_info,
            "game_state": self.game_state.get_public_info(),
        }

        return observation, reward, terminated, truncated, info

    def _get_observation(self, player_id: int) -> np.ndarray:
        """
        Get observation for specific player.

        Args:
            player_id: Player ID

        Returns:
            Observation vector
        """
        player_dice = self.game_state.get_player_dice(player_id)

        observation = create_observation_vector(
            current_bid=self.game_state.current_bid,
            bid_history=self.game_state.bid_history,
            player_dice_count=self.game_state.player_dice_count,
            current_player=self.game_state.current_player,
            palifico_active=self.game_state.palifico_active,
            pacao_called=self.game_state.pacao_called,
            player_dice=player_dice,
            history_length=self.history_length,
            max_players=self.num_players,
            agent_id=player_id,
            num_agents=self.num_players,
        )

        return observation

    def render(self):
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
            print(f"Pacao called: {self.game_state.pacao_called}")
            print(f"Game over: {self.game_state.game_over}")
            if self.game_state.winner is not None:
                print(f"Winner: Player {self.game_state.winner}")
            print("=" * 30)

    def set_active_player(self, player_id: int) -> None:
        """
        Set active player (for whom observation is returned).

        Args:
            player_id: Player ID
        """
        self.active_player_id = player_id

    def get_observation_for_player(self, player_id: int) -> np.ndarray:
        """
        Get observation for specific player.

        Args:
            player_id: Player ID

        Returns:
            Observation vector
        """
        return self._get_observation(player_id)
