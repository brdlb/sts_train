"""
API endpoints for model management.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from pydantic import BaseModel

from ..game_server import GameServer

router = APIRouter(prefix="/api/models", tags=["models"])

# Global game server instance (will be initialized in main.py)
game_server: GameServer = None


def set_game_server(server: GameServer):
    """Set game server instance."""
    global game_server
    game_server = server


class ModelInfo(BaseModel):
    """Model information response."""

    id: str
    path: str
    step: int = None
    elo: float = None
    winrate: float = None
    source: str


@router.get("/list", response_model=List[ModelInfo])
async def list_models():
    """
    Get list of available models.

    Returns:
        List of available models
    """
    if game_server is None:
        raise HTTPException(status_code=500, detail="Game server not initialized")

    models = game_server.get_available_models()
    return [ModelInfo(**model) for model in models]


@router.get("/{model_id}/info", response_model=ModelInfo)
async def get_model_info(model_id: str):
    """
    Get information about a specific model.

    Args:
        model_id: Model ID or path

    Returns:
        Model information
    """
    if game_server is None:
        raise HTTPException(status_code=500, detail="Game server not initialized")

    models = game_server.get_available_models()
    model = next((m for m in models if m["id"] == model_id or m["path"] == model_id), None)

    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    return ModelInfo(**model)


@router.post("/validate")
async def validate_model(model_path: str):
    """
    Validate that a model path exists and is accessible.

    Args:
        model_path: Path to model file

    Returns:
        Validation result
    """
    import os

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    if not model_path.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Model file must be a .zip file")

    return {"valid": True, "path": model_path}

