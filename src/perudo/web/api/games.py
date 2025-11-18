"""
API endpoints for game management.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session
import json

from ..game_server import GameServer
from ..database.database import get_db
from ..database.operations import get_game_history

router = APIRouter(prefix="/api/games", tags=["games"])

# Global game server instance (will be initialized in main.py)
game_server: GameServer = None


def set_game_server(server: GameServer):
    """Set game server instance."""
    global game_server
    game_server = server


class CreateGameRequest(BaseModel):
    """Request model for creating a game."""

    model_paths: List[str]  # Exactly 3 model paths for AI players


class ActionRequest(BaseModel):
    """Request model for game action."""

    action: int  # Action code from action space


@router.post("/create")
async def create_game(request: CreateGameRequest):
    """
    Create a new game.

    Args:
        request: Create game request with model paths

    Returns:
        Game information
    """
    if game_server is None:
        raise HTTPException(status_code=500, detail="Game server not initialized")

    if len(request.model_paths) != 3:
        raise HTTPException(
            status_code=400,
            detail="Must provide exactly 3 model paths for AI players",
        )

    try:
        game_id, session = game_server.create_game(request.model_paths)
        state = session.get_public_state()
        return {
            "game_id": game_id,
            "state": state,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create game: {str(e)}")


@router.get("/{game_id}")
async def get_game_state(game_id: str):
    """
    Get current game state.

    Args:
        game_id: Game session ID

    Returns:
        Game state
    """
    if game_server is None:
        raise HTTPException(status_code=500, detail="Game server not initialized")

    session = game_server.get_game(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")

    # get_public_state() automatically syncs state, so no manual sync needed
    return session.get_public_state()


@router.post("/{game_id}/action")
async def make_action(game_id: str, request: ActionRequest):
    """
    Make an action in the game.

    Args:
        game_id: Game session ID
        request: Action request

    Returns:
        Action result and updated game state
    """
    if game_server is None:
        raise HTTPException(status_code=500, detail="Game server not initialized")

    session = game_server.get_game(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")

    # Make human action
    result = session.make_human_action(request.action)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # _process_action() already syncs state, so no manual sync needed
    # Get updated state
    result["state"] = session.get_public_state()

    return result


@router.post("/{game_id}/continue-round")
async def continue_round(game_id: str):
    """
    Continue to next round after reveal (challenge/believe).
    
    This endpoint is called after the user has viewed the reveal modal
    and is ready to proceed to the next round.
    
    Args:
        game_id: Game session ID
        
    Returns:
        Updated game state
    """
    if game_server is None:
        raise HTTPException(status_code=500, detail="Game server not initialized")
    
    session = game_server.get_game(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")
    
    if not session.awaiting_reveal_confirmation:
        raise HTTPException(
            status_code=400, 
            detail="Not awaiting reveal confirmation"
        )
    
    try:
        # Continue to next round
        session.continue_to_next_round()
        
        # Return updated state
        return {
            "success": True,
            "state": session.get_public_state()
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{game_id}/ai-turns")
async def stream_ai_turns(game_id: str):
    """
    Stream AI turns as Server-Sent Events (SSE).
    
    This endpoint streams each AI turn separately, allowing the client
    to receive updates in real-time as bots make their moves.
    
    Args:
        game_id: Game session ID
        
    Returns:
        SSE stream with AI turn updates
    """
    if game_server is None:
        raise HTTPException(status_code=500, detail="Game server not initialized")

    session = game_server.get_game(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")

    def generate():
        """Generator function for SSE stream."""
        try:
            # Sync state before processing (get_public_state() will sync, but we need to check first)
            session._sync_state()

            # Only process if it's not human's turn and game is not over
            if session.game_over or session.current_player == 0:
                # Send final state
                yield f"data: {json.dumps({'type': 'done', 'state': session.get_public_state()})}\n\n"
                return

            # Process AI turns and stream each one
            for turn_result in session.process_ai_turns_streaming():
                # Format as SSE event
                data = json.dumps({
                    "type": "ai_turn",
                    "player_id": turn_result["player_id"],
                    "action": turn_result["action"],
                    "reward": turn_result["reward"],
                    "state": turn_result["state"],
                    "game_over": turn_result["game_over"],
                    "winner": turn_result.get("winner"),
                })
                yield f"data: {data}\n\n"

            # Send final state when done
            yield f"data: {json.dumps({'type': 'done', 'state': session.get_public_state()})}\n\n"
        except Exception as e:
            # Send error event
            error_data = json.dumps({
                "type": "error",
                "error": str(e),
            })
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
        },
    )


@router.get("/db/{db_game_id}/history")
async def get_game_history_by_db_id(db_game_id: int, db: Session = Depends(get_db)):
    """
    Get game history from database by database game ID.

    Args:
        db_game_id: Database game ID
        db: Database session

    Returns:
        Game history
    """
    from ..database.operations import get_game
    
    game = get_game(db, db_game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found")

    history = get_game_history(db, db_game_id)
    return history


@router.get("/{game_id}/history")
async def get_game_history_endpoint(game_id: str, db: Session = Depends(get_db)):
    """
    Get game history from database.

    Args:
        game_id: Game session ID
        db: Database session

    Returns:
        Game history
    """
    session = game_server.get_game(game_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Game not found")

    history = get_game_history(db, session.db_game_id)
    return history


@router.get("")
async def list_games(
    finished: Optional[bool] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """
    List games with optional filtering.

    Args:
        finished: Filter by finished status
        limit: Maximum number of games to return
        db: Database session

    Returns:
        List of games
    """
    from ..database.models import Game

    query = db.query(Game)

    if finished is not None:
        query = query.filter(Game.is_finished == finished)

    query = query.order_by(Game.created_at.desc()).limit(limit)

    games = query.all()

    return [
        {
            "id": game.id,
            "game_id": None,  # Database ID, not session ID
            "created_at": game.created_at.isoformat() if game.created_at else None,
            "finished_at": game.finished_at.isoformat() if game.finished_at else None,
            "winner": game.winner,
            "num_players": game.num_players,
            "is_finished": game.is_finished,
        }
        for game in games
    ]

