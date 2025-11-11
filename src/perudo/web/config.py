"""
Configuration for web application.
"""

import os
from typing import Optional
from ..training.config import DEFAULT_CONFIG


class WebConfig:
    """Configuration for web application."""

    def __init__(self):
        """Initialize web configuration."""
        # Get paths from training config
        training_config = DEFAULT_CONFIG.training
        game_config = DEFAULT_CONFIG.game

        # Model paths
        self.model_dir = training_config.model_dir
        self.opponent_pool_dir = os.path.join(
            training_config.model_dir, "opponent_pool"
        )

        # Server settings
        self.host: str = os.getenv("WEB_HOST", "0.0.0.0")
        self.port: int = int(os.getenv("WEB_PORT", "8000"))
        self.debug: bool = os.getenv("WEB_DEBUG", "false").lower() == "true"

        # CORS settings
        self.cors_origins: list = [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
        
        # CORS regex pattern for port ranges
        # Allows ports 5100-5199 (e.g., 5173, 5174, etc.) for development servers
        self.cors_origin_regex: str = r"http://(localhost|127\.0\.0\.1):51\d{2}"

        # Database settings
        db_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(db_dir, exist_ok=True)
        self.database_url: str = os.path.join(db_dir, "perudo_games.db")

        # Game settings
        self.default_num_players: int = game_config.num_players
        self.dice_per_player: int = game_config.dice_per_player
        self.total_dice_values: int = game_config.total_dice_values
        self.max_quantity: int = game_config.max_quantity
        self.history_length: int = game_config.history_length

        # Transformer settings (for model loading)
        self.transformer_history_length: int = (
            training_config.transformer_history_length
        )


# Global configuration instance
web_config = WebConfig()

