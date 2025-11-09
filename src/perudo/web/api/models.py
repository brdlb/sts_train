"""
API endpoints for model management.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from ..game_server import GameServer
from ..config import web_config

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
    step: Optional[int] = None
    elo: Optional[float] = None
    winrate: Optional[float] = None
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

    try:
        models = game_server.get_available_models()
        # Convert to ModelInfo, handling any validation errors
        result = []
        for model in models:
            try:
                result.append(ModelInfo(**model))
            except Exception as e:
                # Skip models that don't match the expected schema
                import traceback
                print(f"Warning: Skipping invalid model data: {model}, error: {e}")
                if web_config.debug:
                    traceback.print_exc()
                continue
        return result
    except Exception as e:
        # Return detailed error message
        import traceback
        error_detail = f"Failed to load models: {str(e)}"
        print(f"ERROR: {error_detail}")
        if web_config.debug:
            traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_detail)


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

    try:
        models = game_server.get_available_models()
        model = next((m for m in models if m["id"] == model_id or m["path"] == model_id), None)

        if model is None:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        try:
            return ModelInfo(**model)
        except Exception as e:
            error_detail = f"Invalid model data for {model_id}: {str(e)}"
            print(f"ERROR: {error_detail}")
            if web_config.debug:
                import traceback
                traceback.print_exc()
            raise HTTPException(status_code=500, detail=error_detail)
    except HTTPException:
        # Re-raise HTTP exceptions (like 404)
        raise
    except Exception as e:
        error_detail = f"Failed to get model info: {str(e)}"
        print(f"ERROR: {error_detail}")
        if web_config.debug:
            import traceback
            traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_detail)


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

