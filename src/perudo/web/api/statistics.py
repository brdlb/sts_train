"""
API endpoints for statistics.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from sqlalchemy.orm import Session

from ..database.database import get_db
from ..database.operations import get_player_statistics, get_model_statistics

router = APIRouter(prefix="/api/statistics", tags=["statistics"])


@router.get("/games")
async def get_game_statistics(db: Session = Depends(get_db)):
    """
    Get general game statistics.

    Args:
        db: Database session

    Returns:
        Game statistics
    """
    from ..database.models import Game
    from sqlalchemy import func

    total_games = db.query(Game).count()
    finished_games = db.query(Game).filter(Game.is_finished == True).count()
    active_games = total_games - finished_games

    return {
        "total_games": total_games,
        "finished_games": finished_games,
        "active_games": active_games,
    }


@router.get("/player")
async def get_player_statistics_endpoint(db: Session = Depends(get_db)):
    """
    Get player statistics (human player).

    Args:
        db: Database session

    Returns:
        Player statistics
    """
    stats = get_player_statistics(db)
    return stats


@router.get("/models")
async def get_model_statistics_endpoint(db: Session = Depends(get_db)):
    """
    Get statistics by model.

    Args:
        db: Database session

    Returns:
        Model statistics
    """
    stats = get_model_statistics(db)
    return stats

