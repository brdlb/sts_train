"""
Game server for managing game sessions with human and AI players.
"""

import uuid
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime
import numpy as np
import json
import os
import re
import traceback
import time
import random

from ..game.perudo_env import PerudoEnv
from ..agents.rl_agent import RLAgent
from ..training.opponent_pool import OpponentPool
from ..training.config import DEFAULT_CONFIG
from .database.operations import (
    create_game,
    finish_game,
    add_action,
    save_game_state,
    get_game as get_db_game,
)
from .database.database import SessionLocal
from .config import web_config


def validate_environment_config() -> None:
    """
    Validate that environment configuration matches training configuration.
    
    This ensures that models trained with specific parameters can be loaded
    and used correctly in the server environment.
    
    Raises:
        ValueError: If environment configuration doesn't match training configuration
    """
    training_config = DEFAULT_CONFIG.training
    game_config = DEFAULT_CONFIG.game
    
    # Check max_history_length matches transformer_history_length
    if web_config.transformer_history_length != training_config.transformer_history_length:
        raise ValueError(
            f"Environment max_history_length ({web_config.transformer_history_length}) "
            f"does not match training transformer_history_length ({training_config.transformer_history_length}). "
            f"Models were trained with history_length={training_config.transformer_history_length}."
        )
    
    # Check max_quantity matches
    if web_config.max_quantity != game_config.max_quantity:
        raise ValueError(
            f"Environment max_quantity ({web_config.max_quantity}) "
            f"does not match training max_quantity ({game_config.max_quantity})."
        )
    
    # Check dice_per_player matches
    if web_config.dice_per_player != game_config.dice_per_player:
        raise ValueError(
            f"Environment dice_per_player ({web_config.dice_per_player}) "
            f"does not match training dice_per_player ({game_config.dice_per_player})."
        )
    
    # Check total_dice_values matches
    if web_config.total_dice_values != game_config.total_dice_values:
        raise ValueError(
            f"Environment total_dice_values ({web_config.total_dice_values}) "
            f"does not match training total_dice_values ({game_config.total_dice_values})."
        )
    
    # Check random_num_players matches (important for observation space size)
    # Note: This is checked indirectly through environment creation, but we validate here for clarity
    # The actual observation space size depends on random_num_players and max_players
    if game_config.random_num_players:
        # If random_num_players=True, max_num_players = max(max_players, num_players)
        # Observation space size = 3 * max_num_players + 9
        expected_max_players = max(game_config.max_players, game_config.num_players)
    else:
        # If random_num_players=False, max_num_players = num_players
        # Observation space size = 3 * num_players + 9
        expected_max_players = game_config.num_players
    
    # Calculate expected static_info size
    expected_static_info_size = 3 * expected_max_players + 9
    # Note: We can't check this directly here since environment isn't created yet,
    # but this validation ensures the configuration is correct before environment creation


class GameSession:
    """Game session with human and AI players."""

    def __init__(
        self,
        game_id: str,
        model_paths: List[str],
        db_game_id: int,
    ):
        """
        Initialize game session.

        Args:
            game_id: Unique game session ID
            model_paths: List of model paths for AI players (3 models for 3 AI players)
            db_game_id: Database game ID
        
        Raises:
            ValueError: If environment configuration doesn't match training configuration
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model cannot be loaded due to incompatibility
        """
        self.game_id = game_id
        self.db_game_id = db_game_id
        self.model_paths = model_paths

        # Validate environment configuration before creating environment
        try:
            validate_environment_config()
        except ValueError as e:
            raise ValueError(
                f"Environment configuration mismatch: {str(e)}\n"
                f"Please ensure web_config matches training configuration."
            ) from e

        # Create environment (4 players: human at position 0, 3 AI at positions 1, 2, 3)
        # Environment parameters must match training configuration
        game_config = DEFAULT_CONFIG.game
        self.env = PerudoEnv(
            num_players=4,
            dice_per_player=web_config.dice_per_player,
            total_dice_values=web_config.total_dice_values,
            max_quantity=web_config.max_quantity,
            history_length=web_config.history_length,
            max_history_length=web_config.transformer_history_length,
            random_num_players=game_config.random_num_players,  # Must match training config
            min_players=game_config.min_players,  # Must match training config
            max_players=game_config.max_players,  # Must match training config
            reward_config=DEFAULT_CONFIG.reward,
        )

        # Reset environment
        self.obs, self.info = self.env.reset()

        # Update current player from environment
        self.current_player = self.env.game_state.current_player

        # Create AI agents for players 1, 2, 3
        self.ai_agents: Dict[int, RLAgent] = {}
        for i, model_path in enumerate(model_paths):
            player_id = i + 1  # AI players are at positions 1, 2, 3
            
            # Validate model path exists
            if not os.path.exists(model_path):
                error_msg = (
                    f"Model file not found for player {player_id}: {model_path}\n"
                    f"Please check that the model file exists and the path is correct."
                )
                print(f"ERROR: {error_msg}")
                raise FileNotFoundError(error_msg)
            
            try:
                # Create agent with load_model_later=True to avoid creating temporary model
                # This ensures the model is loaded with the correct architecture from the saved file
                agent = RLAgent(
                    agent_id=player_id,
                    env=self.env,
                    load_model_later=True,  # Don't create model now, will load from file
                )
                
                # Load model with explicit device="cpu" for server usage
                # This avoids GPU overhead and memory issues
                agent.load(model_path, device="cpu")
                self.ai_agents[player_id] = agent
                
                if web_config.debug:
                    print(f"Successfully loaded model for player {player_id}: {model_path}")
                    
            except FileNotFoundError as e:
                # FileNotFoundError is already handled above, but re-raise for clarity
                raise
            except ValueError as e:
                # ValueError from load() indicates model incompatibility
                error_msg = (
                    f"Failed to load model for player {player_id}: {model_path}\n"
                    f"Error: {str(e)}\n"
                    f"This usually means the model was trained with different environment parameters.\n"
                    f"Please ensure:\n"
                    f"  - max_history_length matches transformer_history_length ({web_config.transformer_history_length})\n"
                    f"  - max_quantity matches ({web_config.max_quantity})\n"
                    f"  - observation_space structure matches (Dict with 'bid_history' and 'static_info')\n"
                    f"  - action_space matches\n"
                    f"Full traceback:\n{traceback.format_exc()}"
                )
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg) from e
            except Exception as e:
                # Catch any other unexpected errors
                error_msg = (
                    f"Unexpected error loading model for player {player_id}: {model_path}\n"
                    f"Error type: {type(e).__name__}\n"
                    f"Error message: {str(e)}\n"
                    f"Full traceback:\n{traceback.format_exc()}"
                )
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg) from e

        # Game state tracking
        self.turn_number = 0
        self.game_over = False

        # Extended action history with consequences
        # Each entry contains: player_id, action_type, action_data, consequences
        self.extended_action_history: List[Dict[str, Any]] = []

        # Save initial state to DB
        self._save_state_to_db()

    def get_public_state(self) -> Dict[str, Any]:
        """
        Get public game state (what human player can see).

        Returns:
            Dictionary with game state (JSON-serializable)
        """
        # Sync state from game_state before returning
        # This ensures current_player and game_over are always up-to-date
        self.current_player = self.env.game_state.current_player
        self.game_over = self.env.game_state.game_over

        game_state = self.env.game_state
        public_info = game_state.get_public_info()

        # Get player's own dice observation (contains numpy arrays)
        player_dice_obs = self.env.get_observation_for_player(0)  # Human is player 0
        
        # Convert numpy arrays to Python lists for JSON serialization
        # Exclude action_mask as it's not needed on frontend
        player_dice = {
            "bid_history": player_dice_obs["bid_history"].tolist() if hasattr(player_dice_obs["bid_history"], "tolist") else player_dice_obs["bid_history"],
            "static_info": player_dice_obs["static_info"].tolist() if hasattr(player_dice_obs["static_info"], "tolist") else player_dice_obs["static_info"],
        }
        
        # Also get the actual dice values (not just observation)
        # This is more useful for the frontend to display player's dice
        actual_player_dice = self.env.game_state.get_player_dice(0)
        player_dice["dice_values"] = list(actual_player_dice)
        
        # Convert bid_history tuples to lists for JSON serialization
        bid_history_serializable = [
            self._serialize_value(bid) for bid in game_state.bid_history
        ]
        
        # Convert current_bid tuple to list if it exists
        current_bid_serializable = self._serialize_value(game_state.current_bid)

        # Serialize winner (may be numpy scalar or int)
        winner_serializable = None
        if self.game_over and game_state.winner is not None:
            if hasattr(game_state.winner, "item"):
                winner_serializable = int(game_state.winner.item())
            elif isinstance(game_state.winner, (int, np.integer)):
                winner_serializable = int(game_state.winner)
            else:
                winner_serializable = int(game_state.winner) if game_state.winner is not None else None
        
        # Serialize extended action history
        extended_history_serializable = [
            {
                "player_id": entry["player_id"],
                "action_type": entry["action_type"],
                "action_data": entry["action_data"],
                "consequences": entry["consequences"],
                "turn_number": entry["turn_number"],
            }
            for entry in self.extended_action_history
        ]

        return {
            "game_id": self.game_id,
            "current_player": int(self.current_player),
            "turn_number": int(self.turn_number),
            "game_over": bool(self.game_over),
            "winner": winner_serializable,
            "player_dice_count": self._serialize_value(game_state.player_dice_count),
            "current_bid": current_bid_serializable,
            "bid_history": bid_history_serializable,
            "extended_action_history": extended_history_serializable,
            "palifico_active": self._serialize_value(game_state.palifico_active),
            "believe_called": bool(game_state.believe_called),
            "player_dice": player_dice,  # Only human player's dice (converted to lists)
            "public_info": self._serialize_public_info(public_info),
        }
    
    def _serialize_public_info(self, public_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize public_info dictionary to be JSON-compatible.
        
        Args:
            public_info: Public info dictionary from game_state
            
        Returns:
            JSON-serializable dictionary
        """
        serialized = {}
        for key, value in public_info.items():
            serialized[key] = self._serialize_value(value)
        return serialized
    
    def _serialize_value(self, value: Any) -> Any:
        """
        Recursively serialize a value to be JSON-compatible.
        
        Args:
            value: Value to serialize
            
        Returns:
            JSON-serializable value
        """
        # Handle None
        if value is None:
            return None
        
        # Handle numpy arrays
        if isinstance(value, np.ndarray) or hasattr(value, "tolist"):
            return value.tolist()
        
        # Handle numpy scalars (int, float, bool)
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
        
        # Handle numpy scalar types that have item() method
        if hasattr(value, "item") and not isinstance(value, (list, dict, tuple, str)):
            try:
                return value.item()
            except (ValueError, AttributeError):
                pass
        
        # Handle tuples - convert to lists
        if isinstance(value, tuple):
            return [self._serialize_value(item) for item in value]
        
        # Handle lists - recursively serialize items
        if isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        
        # Handle dictionaries - recursively serialize values
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        
        # Handle basic types - these are already JSON-serializable
        if isinstance(value, (bool, int, float, str)):
            return value
        
        # For any other type, try to convert to string as fallback
        # This should not happen in normal cases, but provides a safe fallback
        return str(value)

    def make_human_action(self, action: int) -> Dict[str, Any]:
        """
        Make action for human player.

        Args:
            action: Action code

        Returns:
            Result dictionary with new state
        """
        if self.game_over:
            return {"error": "Game is over"}

        if self.current_player != 0:
            return {"error": "Not human player's turn"}

        # Set active player to human
        self.env.set_active_player(0)

        # Execute action
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Convert action to readable format
        from ..utils.helpers import action_to_bid
        action_type, param1, param2 = action_to_bid(action, web_config.max_quantity)

        action_data = {
            "action_type": action_type,
            "quantity": param1 if action_type == "bid" else None,
            "value": param2 if action_type == "bid" else None,
        }

        # Extract consequences from info
        consequences = self._extract_consequences(info, 0, action_type)

        # Add to extended action history
        self.extended_action_history.append({
            "player_id": 0,
            "action_type": action_type,
            "action_data": action_data,
            "consequences": consequences,
            "turn_number": self.turn_number,
        })

        # Save action to DB
        add_action(
            SessionLocal(),
            self.db_game_id,
            0,
            action_type,
            action_data,
            self.turn_number,
        )

        self.turn_number += 1

        # CRITICAL: Sync current_player and game_over from game_state after action
        # This ensures we have the correct state after challenge/believe when rounds restart
        self.current_player = self.env.game_state.current_player
        self.game_over = self.env.game_state.game_over or terminated or truncated

        # Save state to DB
        self._save_state_to_db()

        # Serialize reward (may be numpy scalar)
        if hasattr(reward, "item"):
            reward_serializable = float(reward.item())
        elif isinstance(reward, (int, float)):
            reward_serializable = float(reward)
        else:
            reward_serializable = 0.0
        
        result = {
            "success": True,
            "action": action_data,
            "reward": reward_serializable,
            "game_over": self.game_over,
            "winner": int(self.env.game_state.winner) if self.game_over and self.env.game_state.winner is not None else None,
        }

        # If game is over, finish it in DB
        if self.game_over:
            finish_game(SessionLocal(), self.db_game_id, self.env.game_state.winner)

        return result

    def process_ai_turns(self) -> List[Dict[str, Any]]:
        """
        Process all AI turns until it's human's turn or game is over.

        This method follows the same logic as perudo_vec_env.py to ensure
        turn order is consistent with training.

        Turn order:
        - Players take turns in cyclic order: 0, 1, 2, 3, 0, 1, 2, 3...
        - After challenge/believe, the next round starts with the player who lost/gained dice
        - Players with 0 dice are automatically skipped
        - The method uses game_state.current_player as the single source of truth

        Returns:
            List of actions taken by AI players
        """
        actions_taken = []

        # Use game_state.current_player as the single source of truth
        # Sync self.current_player and self.game_over from game_state
        self.current_player = self.env.game_state.current_player
        self.game_over = self.env.game_state.game_over

        # Safety check: validate current_player is within valid range
        if self.current_player < 0 or self.current_player >= self.env.game_state.num_players:
            if web_config.debug:
                print(f"Warning: Invalid current_player {self.current_player}, resetting to 0")
            self.current_player = 0
            self.env.game_state.current_player = 0

        max_steps = 100  # Protection against infinite loops
        step_count = 0

        while not self.game_over and self.current_player != 0 and step_count < max_steps:
            step_count += 1

            # CRITICAL: Use game_state.current_player as the single source of truth
            # Always sync from game_state before making decisions
            self.current_player = self.env.game_state.current_player
            self.game_over = self.env.game_state.game_over

            # Safety check: validate current_player is within valid range
            if self.current_player < 0 or self.current_player >= self.env.game_state.num_players:
                if web_config.debug:
                    print(f"Warning: Invalid current_player {self.current_player} after sync, breaking")
                break

            # Check if game ended after sync
            if self.game_over:
                break

            # Check if it's human's turn (player 0)
            if self.current_player == 0:
                break

            ai_player_id = self.current_player

            # CRITICAL: Skip players with no dice - they have already lost and cannot make moves
            # Players with 0 dice are eliminated and should be automatically skipped
            # This matches the logic in perudo_vec_env.py
            if self.env.game_state.player_dice_count[ai_player_id] == 0:
                # Player has no dice, skip to next player with dice
                # Use next_player() which automatically skips players with 0 dice
                self.env.game_state.next_player()
                self.current_player = self.env.game_state.current_player
                self.game_over = self.env.game_state.game_over
                # Check if game ended after skipping players
                if self.game_over:
                    break
                # Continue to next iteration to check next player
                continue

            # Check if this is an AI player
            if ai_player_id not in self.ai_agents:
                # Not an AI player (shouldn't happen, but handle gracefully)
                break

            # Get observation for AI player
            obs = self.env.get_observation_for_player(ai_player_id)
            self.env.set_active_player(ai_player_id)

            # Add random delay (1-4 seconds) before bot makes a move
            # This makes the game feel more natural when playing with humans
            delay = random.uniform(1.0, 4.0)
            time.sleep(delay)

            # Get action from AI agent
            ai_agent = self.ai_agents[ai_player_id]
            action = ai_agent.act(obs, deterministic=True)

            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Convert action to readable format
            from ..utils.helpers import action_to_bid
            action_type, param1, param2 = action_to_bid(action, web_config.max_quantity)

            action_data = {
                "action_type": action_type,
                "quantity": param1 if action_type == "bid" else None,
                "value": param2 if action_type == "bid" else None,
            }

            # Extract consequences from info
            consequences = self._extract_consequences(info, ai_player_id, action_type)

            # Add to extended action history
            self.extended_action_history.append({
                "player_id": ai_player_id,
                "action_type": action_type,
                "action_data": action_data,
                "consequences": consequences,
                "turn_number": self.turn_number,
            })

            # Save action to DB
            add_action(
                SessionLocal(),
                self.db_game_id,
                ai_player_id,
                action_type,
                action_data,
                self.turn_number,
            )

            # Serialize reward (may be numpy scalar)
            if hasattr(reward, "item"):
                reward_serializable = float(reward.item())
            elif isinstance(reward, (int, float)):
                reward_serializable = float(reward)
            else:
                reward_serializable = 0.0
            
            actions_taken.append({
                "player_id": ai_player_id,
                "action": action_data,
                "reward": reward_serializable,
            })

            self.turn_number += 1

            # CRITICAL: Sync current_player and game_over from game_state after each action
            # This ensures we have the correct state after challenge/believe when rounds restart
            self.current_player = self.env.game_state.current_player
            self.game_over = self.env.game_state.game_over or terminated or truncated

            # Save state to DB
            self._save_state_to_db()

            # If game is over, finish it in DB
            if self.game_over:
                finish_game(SessionLocal(), self.db_game_id, self.env.game_state.winner)

        return actions_taken

    def process_ai_turns_streaming(self):
        """
        Process AI turns one by one, yielding each turn result.
        
        This is a generator that yields each AI turn separately,
        allowing the client to receive updates in real-time.
        
        Yields:
            Dictionary with action result and updated game state
        """
        # Use game_state.current_player as the single source of truth
        # Sync self.current_player and self.game_over from game_state
        self.current_player = self.env.game_state.current_player
        self.game_over = self.env.game_state.game_over

        # Safety check: validate current_player is within valid range
        if self.current_player < 0 or self.current_player >= self.env.game_state.num_players:
            if web_config.debug:
                print(f"Warning: Invalid current_player {self.current_player}, resetting to 0")
            self.current_player = 0
            self.env.game_state.current_player = 0

        max_steps = 100  # Protection against infinite loops
        step_count = 0

        while not self.game_over and self.current_player != 0 and step_count < max_steps:
            step_count += 1

            # CRITICAL: Use game_state.current_player as the single source of truth
            # Always sync from game_state before making decisions
            self.current_player = self.env.game_state.current_player
            self.game_over = self.env.game_state.game_over

            # Safety check: validate current_player is within valid range
            if self.current_player < 0 or self.current_player >= self.env.game_state.num_players:
                if web_config.debug:
                    print(f"Warning: Invalid current_player {self.current_player} after sync, breaking")
                break

            # Check if game ended after sync
            if self.game_over:
                break

            # Check if it's human's turn (player 0)
            if self.current_player == 0:
                break

            ai_player_id = self.current_player

            # CRITICAL: Skip players with no dice - they have already lost and cannot make moves
            # Players with 0 dice are eliminated and should be automatically skipped
            # This matches the logic in perudo_vec_env.py
            if self.env.game_state.player_dice_count[ai_player_id] == 0:
                # Player has no dice, skip to next player with dice
                # Use next_player() which automatically skips players with 0 dice
                self.env.game_state.next_player()
                self.current_player = self.env.game_state.current_player
                self.game_over = self.env.game_state.game_over
                # Check if game ended after skipping players
                if self.game_over:
                    break
                # Continue to next iteration to check next player
                continue

            # Check if this is an AI player
            if ai_player_id not in self.ai_agents:
                # Not an AI player (shouldn't happen, but handle gracefully)
                break

            # Get observation for AI player
            obs = self.env.get_observation_for_player(ai_player_id)
            self.env.set_active_player(ai_player_id)

            # Add random delay (1-4 seconds) before bot makes a move
            # This makes the game feel more natural when playing with humans
            delay = random.uniform(1.0, 4.0)
            time.sleep(delay)

            # Get action from AI agent
            ai_agent = self.ai_agents[ai_player_id]
            action = ai_agent.act(obs, deterministic=True)

            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Convert action to readable format
            from ..utils.helpers import action_to_bid
            action_type, param1, param2 = action_to_bid(action, web_config.max_quantity)

            action_data = {
                "action_type": action_type,
                "quantity": param1 if action_type == "bid" else None,
                "value": param2 if action_type == "bid" else None,
            }

            # Extract consequences from info
            consequences = self._extract_consequences(info, ai_player_id, action_type)

            # Add to extended action history
            self.extended_action_history.append({
                "player_id": ai_player_id,
                "action_type": action_type,
                "action_data": action_data,
                "consequences": consequences,
                "turn_number": self.turn_number,
            })

            # Save action to DB
            add_action(
                SessionLocal(),
                self.db_game_id,
                ai_player_id,
                action_type,
                action_data,
                self.turn_number,
            )

            # Serialize reward (may be numpy scalar)
            if hasattr(reward, "item"):
                reward_serializable = float(reward.item())
            elif isinstance(reward, (int, float)):
                reward_serializable = float(reward)
            else:
                reward_serializable = 0.0

            self.turn_number += 1

            # CRITICAL: Sync current_player and game_over from game_state after each action
            # This ensures we have the correct state after challenge/believe when rounds restart
            self.current_player = self.env.game_state.current_player
            self.game_over = self.env.game_state.game_over or terminated or truncated

            # Save state to DB
            self._save_state_to_db()

            # If game is over, finish it in DB
            if self.game_over:
                finish_game(SessionLocal(), self.db_game_id, self.env.game_state.winner)

            # Yield the result for this turn
            yield {
                "player_id": ai_player_id,
                "action": action_data,
                "reward": reward_serializable,
                "state": self.get_public_state(),
                "game_over": self.game_over,
                "winner": int(self.env.game_state.winner) if self.game_over and self.env.game_state.winner is not None else None,
            }

    def _extract_consequences(self, info: Dict[str, Any], player_id: int, action_type: str) -> Dict[str, Any]:
        """
        Extract consequences of an action from info dictionary.
        
        Args:
            info: Info dictionary from env.step()
            player_id: ID of player who made the action
            action_type: Type of action ('bid', 'challenge', 'believe')
            
        Returns:
            Dictionary with consequences information
        """
        # Get action info from info dictionary (may be in action_info key or directly in info)
        action_info = info.get("action_info", {})
        if not action_info:
            # Fallback to direct fields in info if action_info is not available
            action_info = info
        
        consequences = {
            "action_valid": action_info.get("action_valid", False),
            "dice_lost": None,
            "loser_id": None,
            "challenge_success": None,
            "believe_success": None,
            "actual_count": None,
            "bid_quantity": None,
            "bid_value": None,
            "bidder_id": None,
            "error_msg": action_info.get("error_msg"),
        }
        
        game_state = self.env.game_state
        
        # Extract action-specific consequences
        if action_type == "challenge":
            challenge_success = action_info.get("challenge_success")
            dice_lost = action_info.get("dice_lost", 0)
            
            if challenge_success is not None:
                consequences["challenge_success"] = bool(challenge_success)
                consequences["dice_lost"] = int(dice_lost) if dice_lost is not None else 0
                
                # Get the challenged bid information
                # After challenge, current_bid becomes None, but bid_history still has the last bid
                bidder_id = None
                bid_quantity = None
                bid_value = None
                
                # First, try to get from bid_history (most reliable, as it's updated before challenge)
                if game_state.bid_history:
                    last_bid = game_state.bid_history[-1]
                    if len(last_bid) >= 3:
                        bidder_id = int(last_bid[0])
                        bid_quantity = int(last_bid[1])
                        bid_value = int(last_bid[2])
                
                # If not found, try extended history as fallback
                if bidder_id is None:
                    for entry in reversed(self.extended_action_history):
                        if entry["action_type"] == "bid" and entry["action_data"].get("quantity") is not None:
                            bidder_id = entry["player_id"]
                            bid_quantity = entry["action_data"]["quantity"]
                            bid_value = entry["action_data"]["value"]
                            break
                
                consequences["bidder_id"] = bidder_id
                consequences["bid_quantity"] = bid_quantity
                consequences["bid_value"] = bid_value
                
                # Determine loser: if challenge succeeded (bid was wrong), bidder loses
                # If challenge failed (bid was correct), challenger loses
                # Use loser_id from action_info if available, otherwise determine from challenge_success
                if action_info.get("loser_id") is not None:
                    consequences["loser_id"] = int(action_info["loser_id"])
                elif challenge_success:
                    consequences["loser_id"] = bidder_id
                else:
                    consequences["loser_id"] = player_id
                
                # Get actual count from action_info if available
                if action_info.get("actual_count") is not None:
                    consequences["actual_count"] = int(action_info["actual_count"])
        
        elif action_type == "believe":
            believe_success = action_info.get("believe_success")
            dice_lost = action_info.get("dice_lost", 0)
            
            if believe_success is not None:
                consequences["believe_success"] = bool(believe_success)
                consequences["dice_lost"] = int(dice_lost) if dice_lost is not None else 0
                
                # In believe:
                # - If success: caller gains die (or starts next round), no one loses
                # - If failure: caller loses die
                # Use loser_id from action_info if available
                if action_info.get("loser_id") is not None:
                    consequences["loser_id"] = int(action_info["loser_id"])
                elif not believe_success:
                    consequences["loser_id"] = int(player_id)
                    consequences["dice_lost"] = 1
                else:
                    # Success - caller benefits, no one loses
                    consequences["loser_id"] = None
                    consequences["dice_lost"] = 0
                
                # Get the believed bid information
                bidder_id = None
                bid_quantity = None
                bid_value = None
                
                # After believe, current_bid may still be available or in bid_history
                if game_state.current_bid is not None:
                    bid_quantity = int(game_state.current_bid[0])
                    bid_value = int(game_state.current_bid[1])
                    # Try to find who made this bid from bid_history
                    if game_state.bid_history:
                        last_bid = game_state.bid_history[-1]
                        if len(last_bid) >= 3 and last_bid[1] == bid_quantity and last_bid[2] == bid_value:
                            bidder_id = int(last_bid[0])
                elif game_state.bid_history:
                    # If current_bid is None, get from bid_history
                    last_bid = game_state.bid_history[-1]
                    if len(last_bid) >= 3:
                        bidder_id = int(last_bid[0])
                        bid_quantity = int(last_bid[1])
                        bid_value = int(last_bid[2])
                
                # Fallback to extended history
                if bidder_id is None and bid_quantity is None:
                    for entry in reversed(self.extended_action_history):
                        if entry["action_type"] == "bid" and entry["action_data"].get("quantity") is not None:
                            bidder_id = entry["player_id"]
                            bid_quantity = entry["action_data"]["quantity"]
                            bid_value = entry["action_data"]["value"]
                            break
                
                consequences["bidder_id"] = bidder_id
                consequences["bid_quantity"] = bid_quantity
                consequences["bid_value"] = bid_value
                # Get actual count from action_info if available
                if action_info.get("actual_count") is not None:
                    consequences["actual_count"] = int(action_info["actual_count"])
        
        elif action_type == "bid":
            # For bid, consequences are usually none unless challenged later
            # A bid itself doesn't cause immediate consequences
            consequences["dice_lost"] = 0
            consequences["loser_id"] = None
        
        # Get current dice counts for reference
        consequences["player_dice_count_after"] = [
            int(count) for count in game_state.player_dice_count
        ]
        
        return consequences

    def _save_state_to_db(self):
        """Save current game state to database."""
        game_state = self.env.game_state
        state_json = {
            "current_player": game_state.current_player,
            "player_dice_count": game_state.player_dice_count,
            "current_bid": game_state.current_bid,
            "bid_history": game_state.bid_history,
            "palifico_active": game_state.palifico_active,
            "believe_called": game_state.believe_called,
            "game_over": game_state.game_over,
            "winner": game_state.winner,
        }

        save_game_state(
            SessionLocal(),
            self.db_game_id,
            self.turn_number,
            state_json,
        )


class GameServer:
    """Server for managing multiple game sessions."""

    def __init__(self):
        """Initialize game server."""
        self.sessions: Dict[str, GameSession] = {}
        self.opponent_pool: Optional[OpponentPool] = None

        # Initialize opponent pool if directory exists
        import os
        if os.path.exists(web_config.opponent_pool_dir):
            try:
                self.opponent_pool = OpponentPool(
                    pool_dir=web_config.opponent_pool_dir,
                    max_pool_size=20,
                    min_pool_size=10,
                    keep_best=3,
                    snapshot_freq=50000,
                )
            except Exception as e:
                print(f"Warning: Could not initialize opponent pool: {e}")

    def create_game(self, model_paths: List[str]) -> Tuple[str, GameSession]:
        """
        Create a new game session.

        Args:
            model_paths: List of 3 model paths for AI players

        Returns:
            Tuple (game_id, game_session)
        """
        if len(model_paths) != 3:
            raise ValueError("Must provide exactly 3 model paths for AI players")

        # Create game in database
        players_info = [
            {"player_id": 0, "player_type": "human", "model_path": None},
        ]
        for i, model_path in enumerate(model_paths):
            players_info.append({
                "player_id": i + 1,
                "player_type": "ai",
                "model_path": model_path,
            })

        db_game = create_game(SessionLocal(), num_players=4, players_info=players_info)

        # Create game session
        game_id = str(uuid.uuid4())
        session = GameSession(
            game_id=game_id,
            model_paths=model_paths,
            db_game_id=db_game.id,
        )

        self.sessions[game_id] = session

        # Sync state from game_state
        # This ensures we have the correct current_player
        session.current_player = session.env.game_state.current_player
        session.game_over = session.env.game_state.game_over

        # Don't process AI turns here - client will subscribe to SSE stream
        # This allows each bot turn to be sent separately

        return game_id, session

    def get_game(self, game_id: str) -> Optional[GameSession]:
        """
        Get game session by ID.

        Args:
            game_id: Game session ID

        Returns:
            Game session or None
        """
        return self.sessions.get(game_id)

    def delete_game(self, game_id: str) -> bool:
        """
        Delete game session.

        Args:
            game_id: Game session ID

        Returns:
            True if deleted, False if not found
        """
        if game_id in self.sessions:
            del self.sessions[game_id]
            return True
        return False

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models.

        Returns:
            List of model information dictionaries
        """
        models = []

        # Get models from opponent pool
        if self.opponent_pool:
            try:
                pool_snapshots = self.opponent_pool.list_snapshots()
                for snapshot_id, snapshot_info in pool_snapshots.items():
                    try:
                        # Verify that the snapshot path exists
                        if os.path.exists(snapshot_info.get("path", "")):
                            models.append({
                                "id": snapshot_id,
                                "path": snapshot_info["path"],
                                "step": snapshot_info.get("step"),
                                "elo": snapshot_info.get("elo", 1500.0),
                                "winrate": snapshot_info.get("winrate", 0.5),
                                "source": "opponent_pool",
                            })
                    except Exception as e:
                        # Skip individual snapshot if there's an error
                        if web_config.debug:
                            print(f"Warning: Skipping snapshot {snapshot_id}: {e}")
                        continue
            except Exception as e:
                # Log error but continue to try loading from main directory
                error_msg = f"Error loading opponent pool snapshots: {e}"
                print(f"ERROR: {error_msg}")
                if web_config.debug:
                    traceback.print_exc()

        # Get models from main model directory
        import glob
        try:
            if os.path.exists(web_config.model_dir):
                pattern = os.path.join(web_config.model_dir, "*.zip")
                try:
                    for model_path in glob.glob(pattern):
                        try:
                            model_name = os.path.basename(model_path)
                            # Skip opponent pool models (already added)
                            if "opponent_pool" not in model_path:
                                # Verify file exists and is readable
                                if os.path.isfile(model_path):
                                    # Try to extract step from filename
                                    # Pattern: perudo_model_100000_steps.zip -> step=100000
                                    step = None
                                    step_match = re.search(r'_(\d+)_steps\.zip$', model_name)
                                    if step_match:
                                        try:
                                            step = int(step_match.group(1))
                                        except ValueError:
                                            pass
                                    
                                    models.append({
                                        "id": model_name,
                                        "path": model_path,
                                        "step": step,
                                        "elo": None,
                                        "winrate": None,
                                        "source": "main_dir",
                                    })
                        except Exception as e:
                            # Skip individual model if there's an error
                            if web_config.debug:
                                print(f"Warning: Skipping model {model_path}: {e}")
                            continue
                except Exception as e:
                    error_msg = f"Error scanning model directory {web_config.model_dir}: {e}"
                    print(f"ERROR: {error_msg}")
                    if web_config.debug:
                        traceback.print_exc()
        except Exception as e:
            error_msg = f"Error accessing model directory {web_config.model_dir}: {e}"
            print(f"ERROR: {error_msg}")
            if web_config.debug:
                traceback.print_exc()

        return models

