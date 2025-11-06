"""
Game server for managing game sessions with human and AI players.
"""

import uuid
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime
import numpy as np
import json

from ..game.perudo_env import PerudoEnv
from ..agents.rl_agent import RLAgent
from ..training.opponent_pool import OpponentPool
from .database.operations import (
    create_game,
    finish_game,
    add_action,
    save_game_state,
    get_game as get_db_game,
)
from .database.database import SessionLocal
from .config import web_config


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
        """
        self.game_id = game_id
        self.db_game_id = db_game_id
        self.model_paths = model_paths

        # Create environment (4 players: human at position 0, 3 AI at positions 1, 2, 3)
        self.env = PerudoEnv(
            num_players=4,
            dice_per_player=web_config.dice_per_player,
            total_dice_values=web_config.total_dice_values,
            max_quantity=web_config.max_quantity,
            history_length=web_config.history_length,
            max_history_length=web_config.transformer_history_length,
        )

        # Reset environment
        self.obs, self.info = self.env.reset()

        # Update current player from environment
        self.current_player = self.env.game_state.current_player

        # Create AI agents for players 1, 2, 3
        self.ai_agents: Dict[int, RLAgent] = {}
        for i, model_path in enumerate(model_paths):
            player_id = i + 1  # AI players are at positions 1, 2, 3
            try:
                agent = RLAgent(
                    agent_id=player_id,
                    env=self.env,
                    model=None,  # Will load from path
                )
                agent.load(model_path)
                self.ai_agents[player_id] = agent
            except Exception as e:
                print(f"Error loading model {model_path} for player {player_id}: {e}")
                # Create a dummy agent if model fails to load
                # In production, you might want to handle this differently
                raise

        # Game state tracking
        self.turn_number = 0
        self.game_over = False

        # Save initial state to DB
        self._save_state_to_db()

    def get_public_state(self) -> Dict[str, Any]:
        """
        Get public game state (what human player can see).

        Returns:
            Dictionary with game state
        """
        game_state = self.env.game_state
        public_info = game_state.get_public_info()

        # Get player's own dice
        player_dice = self.env.get_observation_for_player(0)  # Human is player 0

        return {
            "game_id": self.game_id,
            "current_player": self.current_player,
            "turn_number": self.turn_number,
            "game_over": self.game_over,
            "winner": game_state.winner if self.game_over else None,
            "player_dice_count": game_state.player_dice_count,
            "current_bid": game_state.current_bid,
            "bid_history": game_state.bid_history,
            "palifico_active": game_state.palifico_active,
            "believe_called": game_state.believe_called,
            "player_dice": player_dice,  # Only human player's dice
            "public_info": public_info,
        }

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
        self.current_player = self.env.game_state.current_player
        self.game_over = terminated or truncated

        # Save state to DB
        self._save_state_to_db()

        result = {
            "success": True,
            "action": action_data,
            "reward": reward,
            "game_over": self.game_over,
            "winner": self.env.game_state.winner if self.game_over else None,
        }

        # If game is over, finish it in DB
        if self.game_over:
            finish_game(SessionLocal(), self.db_game_id, self.env.game_state.winner)

        return result

    def process_ai_turns(self) -> List[Dict[str, Any]]:
        """
        Process all AI turns until it's human's turn or game is over.

        Returns:
            List of actions taken by AI players
        """
        actions_taken = []

        while not self.game_over and self.current_player != 0:
            ai_player_id = self.current_player
            if ai_player_id not in self.ai_agents:
                break

            # Get observation for AI player
            obs = self.env.get_observation_for_player(ai_player_id)
            self.env.set_active_player(ai_player_id)

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

            # Save action to DB
            add_action(
                SessionLocal(),
                self.db_game_id,
                ai_player_id,
                action_type,
                action_data,
                self.turn_number,
            )

            actions_taken.append({
                "player_id": ai_player_id,
                "action": action_data,
                "reward": reward,
            })

            self.turn_number += 1
            self.current_player = self.env.game_state.current_player
            self.game_over = terminated or truncated

            # Save state to DB
            self._save_state_to_db()

            # If game is over, finish it in DB
            if self.game_over:
                finish_game(SessionLocal(), self.db_game_id, self.env.game_state.winner)

        return actions_taken

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

        # Process initial AI turns if needed
        if session.current_player != 0:
            session.process_ai_turns()

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
            pool_snapshots = self.opponent_pool.list_snapshots()
            for snapshot_id, snapshot_info in pool_snapshots.items():
                models.append({
                    "id": snapshot_id,
                    "path": snapshot_info["path"],
                    "step": snapshot_info["step"],
                    "elo": snapshot_info.get("elo", 1500.0),
                    "winrate": snapshot_info.get("winrate", 0.5),
                    "source": "opponent_pool",
                })

        # Get models from main model directory
        import os
        import glob
        if os.path.exists(web_config.model_dir):
            pattern = os.path.join(web_config.model_dir, "*.zip")
            for model_path in glob.glob(pattern):
                model_name = os.path.basename(model_path)
                # Skip opponent pool models (already added)
                if "opponent_pool" not in model_path:
                    models.append({
                        "id": model_name,
                        "path": model_path,
                        "step": None,
                        "elo": None,
                        "winrate": None,
                        "source": "main_dir",
                    })

        return models

