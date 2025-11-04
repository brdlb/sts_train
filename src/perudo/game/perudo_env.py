"""
Gymnasium environment for Perudo game.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Any
import sys
import logging
import os

from .game_state import GameState
from .rules import PerudoRules
from ..utils.helpers import (
    create_observation_vector,
    calculate_reward,
    action_to_bid,
    decode_bid,
)

# Setup debug logger
_debug_logger = None

def get_debug_logger():
    """Get or create logger for episode results only."""
    global _debug_logger
    if _debug_logger is None:
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger for episode results
        _debug_logger = logging.getLogger("perudo_debug")
        _debug_logger.setLevel(logging.INFO)  # Only log INFO level (episode results)
        
        # Clear existing handlers
        _debug_logger.handlers.clear()
        
        # File handler - only INFO level and above
        log_file = os.path.join(log_dir, "game_debug.log")
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        _debug_logger.addHandler(file_handler)
        
        # Console handler - only INFO level and above
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        _debug_logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        _debug_logger.propagate = False
    
    return _debug_logger


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

        self.episode_reward = 0.0
        self.episode_length = 0
        self.last_episode_reward = 0.0
        self.last_episode_length = 0

        # Track round information for intermediate rewards
        # Agent dice count at the start of current round
        self.agent_dice_at_round_start = dice_per_player
        # Track if agent made a bid this round (for successful bluff detection)
        self.agent_bid_this_round = False
        # Deferred reward for agent (e.g., when opponent challenges agent's bid and fails)
        self.agent_deferred_reward = 0.0

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

        self.episode_reward = 0.0
        self.episode_length = 0

        # Initialize round tracking
        self.agent_dice_at_round_start = self.game_state.player_dice_count[0]
        self.agent_bid_this_round = False
        self.agent_deferred_reward = 0.0

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
            # If game is already over, ensure episode statistics are saved
            # This can happen if game ended on opponent's turn in vec_env
            if self.last_episode_reward == 0.0 and self.last_episode_length == 0:
                # Episode stats not saved yet, save current accumulated values
                self.last_episode_reward = self.episode_reward
                self.last_episode_length = self.episode_length
            info = {
                "game_over": True,
                "winner": self.game_state.winner,
                "episode": {
                    "r": self.last_episode_reward,
                    "l": self.last_episode_length,
                },
                "episode_reward": self.last_episode_reward,
                "episode_length": self.last_episode_length,
            }
            return observation, 0.0, True, False, info

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
                    # Track that agent made a bid this round
                    if self.active_player_id == 0:
                        self.agent_bid_this_round = True
                    self.game_state.next_player()
                    reward = calculate_reward(
                        "bid", False, -1, self.active_player_id, dice_lost=0
                    )
            else:
                # Invalid action - penalty
                reward = -1.0
                self.game_state.next_player()
                # Check if game ended after next_player (e.g., only one player left)
                if self.game_state.game_over:
                    # Recalculate reward considering game outcome
                    reward = calculate_reward(
                        "bid",
                        self.game_state.game_over,
                        self.game_state.winner or -1,
                        self.active_player_id,
                        dice_lost=0,
                    )
                action_valid = False

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
                
                # Check if agent (player 0) made the bid and successfully defended it
                # (opponent challenged and failed)
                agent_id = 0
                agent_defended_bid = False
                if self.game_state.bid_history:
                    bid_maker_id = self.game_state.bid_history[-1][0]
                    # Agent made the bid, challenge failed (opponent was wrong), agent didn't lose dice
                    if (bid_maker_id == agent_id and 
                        not challenge_success and 
                        loser_id != agent_id):
                        agent_defended_bid = True
                
                # Check round reward BEFORE losing dice
                # Pass action_type to avoid double penalty (challenge already penalized in calculate_reward)
                round_reward = self._check_round_end_reward(loser_id, dice_lost, action_type="challenge")
                self.game_state.lose_dice(loser_id, dice_lost)
                self.game_state.current_bid = None
                self.game_state.pacao_called = False

                # Restart round from new player
                if loser_id == self.active_player_id:
                    self.game_state.next_player()
                else:
                    self.game_state.current_player = (loser_id + 1) % self.num_players
                    self.game_state.next_player()

                # Roll dice again - round ends
                self.game_state.roll_dice()
                # Reset round tracking after dice roll
                self.agent_dice_at_round_start = self.game_state.player_dice_count[0]
                self.agent_bid_this_round = False

                reward = calculate_reward(
                    "challenge",
                    self.game_state.game_over,
                    self.game_state.winner or -1,
                    self.active_player_id,
                    challenge_success=challenge_success,
                    dice_lost=dice_lost if loser_id == self.active_player_id else 0,
                )
                # Add round reward
                if self.active_player_id == 0:
                    reward += round_reward
                # If agent successfully defended their bid (opponent challenged and failed),
                # store deferred reward for agent (will be given on agent's next turn)
                elif agent_defended_bid:
                    # Agent gets +5.0 reward for successfully defending their risky bid
                    # This reward will be added when agent makes their next action
                    self.agent_deferred_reward += 5.0
                action_valid = True
            else:
                # Invalid challenge - penalty
                reward = -1.0
                self.game_state.next_player()
                # Check if game ended after next_player
                if self.game_state.game_over:
                    reward = calculate_reward(
                        "challenge",
                        self.game_state.game_over,
                        self.game_state.winner or -1,
                        self.active_player_id,
                        challenge_success=False,
                        dice_lost=0,
                    )

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
                
                # Check if agent (player 0) made the bid and successfully defended it
                # (opponent called pacao and failed)
                agent_id = 0
                agent_defended_bid = False
                if self.game_state.bid_history:
                    bid_maker_id = self.game_state.bid_history[-1][0]
                    # Agent made the bid, pacao failed (opponent was wrong), agent didn't lose dice
                    if (bid_maker_id == agent_id and 
                        not pacao_success and 
                        loser_id != agent_id):
                        agent_defended_bid = True
                
                # Check round reward BEFORE losing dice
                # Pass action_type to avoid double penalty (pacao already penalized in calculate_reward)
                round_reward = self._check_round_end_reward(loser_id, dice_lost, action_type="pacao")
                self.game_state.lose_dice(loser_id, dice_lost)
                self.game_state.current_bid = None
                self.game_state.pacao_called = False

                # Restart round
                if loser_id == self.active_player_id:
                    self.game_state.next_player()
                else:
                    self.game_state.current_player = (loser_id + 1) % self.num_players
                    self.game_state.next_player()

                # Roll dice again - round ends
                self.game_state.roll_dice()
                # Reset round tracking after dice roll
                self.agent_dice_at_round_start = self.game_state.player_dice_count[0]
                self.agent_bid_this_round = False

                reward = calculate_reward(
                    "pacao",
                    self.game_state.game_over,
                    self.game_state.winner or -1,
                    self.active_player_id,
                    pacao_success=pacao_success,
                    dice_lost=dice_lost if loser_id == self.active_player_id else 0,
                )
                # Add round reward
                if self.active_player_id == 0:
                    reward += round_reward
                # If agent successfully defended their bid (opponent called pacao and failed),
                # store deferred reward for agent (will be given on agent's next turn)
                elif agent_defended_bid:
                    # Agent gets +5.0 reward for successfully defending their risky bid
                    # This reward will be added when agent makes their next action
                    self.agent_deferred_reward += 5.0
                action_valid = True
            else:
                # Invalid pacao - penalty
                reward = -1.0
                self.game_state.next_player()
                # Check if game ended after next_player
                if self.game_state.game_over:
                    reward = calculate_reward(
                        "pacao",
                        self.game_state.game_over,
                        self.game_state.winner or -1,
                        self.active_player_id,
                        pacao_success=False,
                        dice_lost=0,
                    )

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

        # накопление суммарной награды и длины эпизода
        # ВАЖНО: накапливаем статистику только для learning agent (player_id=0)
        # В vec_env только learning agent делает ходы через этот метод для сбора статистики
        # Оппоненты играют отдельно и их награды не должны учитываться в статистике learning agent
        if self.active_player_id == 0:
            # Add any deferred reward for agent (e.g., from successfully defending bid)
            if self.agent_deferred_reward != 0.0:
                reward += self.agent_deferred_reward
                self.agent_deferred_reward = 0.0
            self.episode_reward += reward
            self.episode_length += 1

        # Check game over
        terminated = self.game_state.game_over
        truncated = False  # Not implemented yet
        done = terminated or truncated
        
        # If episode ended and there's deferred reward for agent, add it to episode reward
        # (This handles the case where episode ends before agent's next turn)
        if done and self.agent_deferred_reward != 0.0 and self.active_player_id != 0:
            # Episode ended on opponent's turn, but agent had deferred reward
            # Add it to episode reward directly
            self.episode_reward += self.agent_deferred_reward
            self.agent_deferred_reward = 0.0
        
        # Log episode results only
        if done:
            logger = get_debug_logger()
            logger.info(f"EPISODE ENDED - Winner: Player {self.game_state.winner}, Learning Agent Reward: {self.episode_reward:.2f}, Length: {self.episode_length}")

        info = {
            "player_id": self.active_player_id,
            "game_over": terminated,
            "winner": self.game_state.winner,
            "action_info": self.last_action_info,
            "game_state": self.game_state.get_public_info(),
        }

        # если эпизод завершился, сохраняем статистику для realtime
        if done:
            self.last_episode_reward = self.episode_reward
            self.last_episode_length = self.episode_length
            # VecMonitor expects "episode" key with "r" (reward) and "l" (length) subkeys
            info["episode"] = {
                "r": self.last_episode_reward,
                "l": self.last_episode_length,
            }
            # Also keep old format for backward compatibility with custom callbacks
            info["episode_reward"] = self.last_episode_reward
            info["episode_length"] = self.last_episode_length
            if hasattr(self.game_state, "winner"):
                info["winner"] = self.game_state.winner

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

    def _check_round_end_reward(self, loser_id: int, dice_lost: int, action_type: Optional[str] = None) -> float:
        """
        Check and return reward for round end.

        Args:
            loser_id: ID of player who lost dice
            dice_lost: Number of dice lost
            action_type: Type of action that ended the round ('challenge', 'pacao', or None)
                        Used to avoid double penalty when agent's challenge/pacao fails

        Returns:
            Round reward (0.1 if agent didn't lose dice, 0 otherwise)
            Also handles -0.5 for unsuccessful bid that led to dice loss
        """
        reward = 0.0
        agent_id = 0

        # Check if agent (player 0) lost dice in this round
        agent_lost_dice = (loser_id == agent_id and dice_lost > 0)

        # -0.5 for unsuccessful bid that led to dice loss
        # (agent made a bid that was successfully challenged by opponent)
        # Only apply if the current action is NOT agent's challenge/pacao
        # (because those are already penalized in calculate_reward)
        if agent_lost_dice and self.agent_bid_this_round:
            # Check if this is agent's own challenge/pacao (already penalized in calculate_reward)
            is_agent_action = (action_type in ["challenge", "pacao"] and self.active_player_id == agent_id)
            if not is_agent_action:
                # Agent's bid was successfully challenged by opponent
                reward -= 0.5

        # +0.1 reward for each round in which the agent did not lose a die
        if not agent_lost_dice:
            reward += 0.1
            
            # +0.5 for successful bluff (agent made a bid that wasn't successfully challenged)
            if self.agent_bid_this_round:
                # Agent successfully bluffed (bid was correct or never challenged)
                reward += 0.5

        return reward
