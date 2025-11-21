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


class PerudoJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for Perudo game objects."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except (ValueError, AttributeError):
                pass
        return super().default(obj)

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
        
        if web_config.debug:
            print(f"Initializing GameSession {game_id} with models: {model_paths}")

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
            auto_advance_round=False,  # Web version requires manual confirmation before advancing to next round
        )

        # Reset environment
        self.obs, self.info = self.env.reset()

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
        
        # Extended action history with consequences
        # Each entry contains: player_id, action_type, action_data, consequences
        self.extended_action_history: List[Dict[str, Any]] = []

        # Flag to track if we're awaiting user confirmation to continue to next round after reveal
        self.awaiting_reveal_confirmation = False

        # Save initial state to DB
        self._save_state_to_db()

    @property
    def current_player(self) -> int:
        """Get current player from environment."""
        return self.env.game_state.current_player

    @property
    def game_over(self) -> bool:
        """Get game over status from environment."""
        return self.env.game_state.game_over

    def _is_human_eliminated(self) -> bool:
        """
        Check if human player (player 0) has been eliminated.
        
        Returns:
            True if human player has no dice left, False otherwise
        """
        return self.env.game_state.player_dice_count[0] == 0

    def get_public_state(self) -> Dict[str, Any]:
        """
        Get public game state (what human player can see).

        Returns:
            Dictionary with game state (JSON-serializable)
        """
        game_state = self.env.game_state
        public_info = game_state.get_public_info()

        # Get player's own dice observation (contains numpy arrays)
        player_dice_obs = self.env.get_observation_for_player(0)  # Human is player 0
        
        # Convert numpy arrays to Python lists for JSON serialization
        # Exclude action_mask as it's not needed on frontend
        player_dice = {
            "bid_history": player_dice_obs["bid_history"],
            "static_info": player_dice_obs["static_info"],
        }
        
        # Also get the actual dice values (not just observation)
        # This is more useful for the frontend to display player's dice
        actual_player_dice = self.env.game_state.get_player_dice(0)
        player_dice["dice_values"] = list(actual_player_dice)
        
        # Convert bid_history to frontend format [player_id, quantity, value]
        # Use extended_action_history to get player_id for each bid
        bid_history_frontend = []
        for entry in self.extended_action_history:
            if entry["action_type"] == "bid" and entry["action_data"].get("quantity") is not None:
                bid_history_frontend.append([
                    entry["player_id"],
                    entry["action_data"]["quantity"],
                    entry["action_data"]["value"]
                ])
        
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

        # Use custom encoder for serialization
        state = {
            "game_id": self.game_id,
            "current_player": int(self.current_player),
            "turn_number": int(self.turn_number),
            "game_over": bool(self.game_over),
            "winner": int(game_state.winner) if game_state.winner is not None else None,
            "player_dice_count": game_state.player_dice_count,
            "current_bid": game_state.current_bid,
            "bid_history": bid_history_frontend,
            "extended_action_history": extended_history_serializable,
            "palifico_active": game_state.palifico_active,
            "believe_called": bool(game_state.believe_called),
            "last_bid_player_id": game_state.last_bid_player_id,
            "player_dice": player_dice,  # Only human player's dice
            "public_info": public_info,
            "awaiting_reveal_confirmation": bool(self.awaiting_reveal_confirmation),
        }
        
        # We return the dict, but the caller should use PerudoJSONEncoder when dumping to JSON
        # For now, we manually convert numpy types in the dict to ensure compatibility
        # with existing callers that might not use the encoder
        return json.loads(json.dumps(state, cls=PerudoJSONEncoder))

    def _process_action(
        self,
        player_id: int,
        action: int,
        reward: Any,
        terminated: bool,
        truncated: bool,
        info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process an action and update game state.
        
        This method handles the common logic for processing any action:
        - Convert action to readable format
        - Extract consequences
        - Save to extended action history
        - Save to database
        - Save state to database
        - Finish game if needed
        
        Args:
            player_id: ID of player who made the action
            action: Action code
            reward: Reward from environment
            terminated: Whether environment was terminated
            truncated: Whether environment was truncated
            info: Info dictionary from environment
            
        Returns:
            Dictionary with action result
        """
        # Convert action to readable format
        from ..utils.helpers import action_to_bid
        action_type, param1, param2 = action_to_bid(action, web_config.max_quantity)
        
        action_data = {
            "action_type": action_type,
            "quantity": param1 if action_type == "bid" else None,
            "value": param2 if action_type == "bid" else None,
        }
        
        # Extract consequences from info
        consequences = self._extract_consequences(info, player_id, action_type)
        
        # Add to extended action history
        self.extended_action_history.append({
            "player_id": player_id,
            "action_type": action_type,
            "action_data": action_data,
            "consequences": consequences,
            "turn_number": self.turn_number,
        })
        
        # Save action to DB
        add_action(
            SessionLocal(),
            self.db_game_id,
            player_id,
            action_type,
            action_data,
            self.turn_number,
        )
        
        # Increment turn number
        self.turn_number += 1
        
        # Check if round advance is needed (for challenge/believe with auto_advance_round=False)
        if info.get("needs_round_advance") and not self.game_over:
            self.awaiting_reveal_confirmation = True
        
        # Save state to DB
        self._save_state_to_db()
        
        # If game is over, finish it in DB
        if self.game_over:
            finish_game(SessionLocal(), self.db_game_id, self.env.game_state.winner)
        
        # Use custom encoder for serialization via json.loads(json.dumps(...))
        # This ensures all numpy types are converted to native Python types
        result = {
            "player_id": player_id,
            "action": action_data,
            "reward": reward,
            "game_over": self.game_over,
            "winner": int(self.env.game_state.winner) if self.game_over and self.env.game_state.winner is not None else None,
        }
        return json.loads(json.dumps(result, cls=PerudoJSONEncoder))

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

        # Process action using common method
        result = self._process_action(0, action, reward, terminated, truncated, info)
        
        # Add success flag for human actions
        result["success"] = True
        
        return result

    def _process_ai_turn_loop(self, streaming: bool = False):
        """
        Common loop for processing AI turns.
        
        This method processes all AI turns until it's human's turn or game is over.
        It always works as a generator, yielding results one by one.
        
        Args:
            streaming: If True, include full state in results; if False, minimal results
            
        Yields:
            Dictionary with action result (and updated game state if streaming=True)
        """
        max_steps = 100  # Protection against infinite loops
        step_count = 0
        
        while not self.game_over and self.current_player != 0 and step_count < max_steps:
            step_count += 1
            
            # Check if it's human's turn (player 0)
            if self.current_player == 0:
                break
            
            ai_player_id = self.current_player
            
            # Check if this is an AI player
            if ai_player_id not in self.ai_agents:
                # Not an AI player (shouldn't happen, but handle gracefully)
                break
            
            # Get observation for AI player
            obs = self.env.get_observation_for_player(ai_player_id)
            self.env.set_active_player(ai_player_id)
            
            # Add delay before bot makes a move
            # Use shorter delay (0.5s) if human player is eliminated, normal delay (1-4s) otherwise
            if self._is_human_eliminated():
                delay = 0.5
            else:
                delay = random.uniform(1.0, 3.0)
            time.sleep(delay)
            
            # Get action from AI agent
            ai_agent = self.ai_agents[ai_player_id]
            action = ai_agent.act(obs, deterministic=True)
            
            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Process action using common method
            result = self._process_action(ai_player_id, action, reward, terminated, truncated, info)
            
            # Add state to result for streaming mode
            if streaming:
                result["state"] = self.get_public_state()
            
            yield result

    def process_ai_turns(self) -> List[Dict[str, Any]]:
        """
        Process all AI turns until it's human's turn or game is over.

        This method follows the same logic as perudo_vec_env.py to ensure
        turn order is consistent with training.

        Returns:
            List of actions taken by AI players
        """
        # Collect results from generator into list
        results = list(self._process_ai_turn_loop(streaming=False))
        # Extract only essential fields for non-streaming mode
        return [
            {
                "player_id": result["player_id"],
                "action": result["action"],
                "reward": result["reward"],
            }
            for result in results
        ]

    def process_ai_turns_streaming(self):
        """
        Process AI turns one by one, yielding each turn result.
        
        This is a generator that yields each AI turn separately,
        allowing the client to receive updates in real-time.
        
        Yields:
            Dictionary with action result and updated game state
        """
        yield from self._process_ai_turn_loop(streaming=True)

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
            consequences["challenge_success"] = action_info.get("challenge_success")
            consequences["dice_lost"] = action_info.get("dice_lost", 0)
            consequences["actual_count"] = action_info.get("actual_count")
            consequences["loser_id"] = action_info.get("loser_id")
            
            # Get the challenged bid information
            # After challenge, current_bid becomes None, but bid_history still has the last bid
            # Or we can use last_bid_player_id from game_state if available
            
            # Try to get from bid_history (most reliable)
            if game_state.bid_history:
                last_bid = game_state.bid_history[-1]
                if len(last_bid) >= 3:
                    consequences["bidder_id"] = int(last_bid[0])
                    consequences["bid_quantity"] = int(last_bid[1])
                    consequences["bid_value"] = int(last_bid[2])
            
            # Fallback to extended history if needed
            if consequences["bidder_id"] is None:
                for entry in reversed(self.extended_action_history):
                    if entry["action_type"] == "bid" and entry["action_data"].get("quantity") is not None:
                        consequences["bidder_id"] = entry["player_id"]
                        consequences["bid_quantity"] = entry["action_data"]["quantity"]
                        consequences["bid_value"] = entry["action_data"]["value"]
                        break
        
        elif action_type == "believe":
            consequences["believe_success"] = action_info.get("believe_success")
            consequences["dice_lost"] = action_info.get("dice_lost", 0)
            consequences["actual_count"] = action_info.get("actual_count")
            consequences["loser_id"] = action_info.get("loser_id")
            
            # Get the believed bid information
            # After believe, current_bid may still be available or in bid_history
            if game_state.current_bid is not None:
                consequences["bid_quantity"] = int(game_state.current_bid[0])
                consequences["bid_value"] = int(game_state.current_bid[1])
                # Try to find who made this bid from bid_history
                if game_state.bid_history:
                    last_bid = game_state.bid_history[-1]
                    if len(last_bid) >= 3 and last_bid[1] == consequences["bid_quantity"] and last_bid[2] == consequences["bid_value"]:
                        consequences["bidder_id"] = int(last_bid[0])
            elif game_state.bid_history:
                # If current_bid is None, get from bid_history
                last_bid = game_state.bid_history[-1]
                if len(last_bid) >= 3:
                    consequences["bidder_id"] = int(last_bid[0])
                    consequences["bid_quantity"] = int(last_bid[1])
                    consequences["bid_value"] = int(last_bid[2])
            
            # Fallback to extended history
            if consequences["bidder_id"] is None:
                for entry in reversed(self.extended_action_history):
                    if entry["action_type"] == "bid" and entry["action_data"].get("quantity") is not None:
                        consequences["bidder_id"] = entry["player_id"]
                        if consequences["bid_quantity"] is None:
                            consequences["bid_quantity"] = entry["action_data"]["quantity"]
                            consequences["bid_value"] = entry["action_data"]["value"]
                        break
        
        elif action_type == "bid":
            # For bid, consequences are usually none unless challenged later
            consequences["dice_lost"] = 0
            consequences["loser_id"] = None
        
        # Get current dice counts for reference
        consequences["player_dice_count_after"] = [
            int(count) for count in game_state.player_dice_count
        ]
        
        # Add all player dice values for reveal (challenge/believe only)
        # This allows frontend to show all dice during reveal phase
        if action_type in ("challenge", "believe"):
            all_player_dice = []
            for player_id in range(game_state.num_players):
                player_dice = game_state.get_player_dice(player_id)
                all_player_dice.append([int(die) for die in player_dice])
            consequences["all_player_dice"] = all_player_dice
        
        return consequences

    def continue_to_next_round(self) -> None:
        """
        Continue to the next round after reveal confirmation.
        
        This method should be called when the user has viewed the reveal modal
        and is ready to proceed to the next round.
        
        Raises:
            RuntimeError: If not awaiting reveal confirmation
        """
        if not self.awaiting_reveal_confirmation:
            raise RuntimeError("Not awaiting reveal confirmation")
        
        # Advance to next round in environment
        self.env.advance_to_next_round()
        
        # Clear the awaiting flag
        self.awaiting_reveal_confirmation = False
        
        # Save state to DB
        self._save_state_to_db()

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
            
        if web_config.debug:
            print(f"GameServer.create_game called with models: {model_paths}")

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

