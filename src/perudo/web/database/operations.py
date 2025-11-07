"""
Database CRUD operations for games.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime
from .models import Game, GamePlayer, GameAction, GameState


def create_game(
    db: Session,
    num_players: int = 4,
    players_info: Optional[List[Dict[str, Any]]] = None,
) -> Game:
    """
    Create a new game.

    Args:
        db: Database session
        num_players: Number of players
        players_info: List of player info dicts with 'player_id', 'player_type', 'model_path'

    Returns:
        Created game
    """
    game = Game(num_players=num_players, is_finished=False)
    db.add(game)
    db.flush()  # Get game.id

    # Create players
    if players_info:
        for player_info in players_info:
            player = GamePlayer(
                game_id=game.id,
                player_id=player_info["player_id"],
                player_type=player_info["player_type"],
                model_path=player_info.get("model_path"),
            )
            db.add(player)

    db.commit()
    db.refresh(game)
    return game


def get_game(db: Session, game_id: int) -> Optional[Game]:
    """
    Get game by ID.

    Args:
        db: Database session
        game_id: Game ID

    Returns:
        Game or None
    """
    return db.query(Game).filter(Game.id == game_id).first()


def finish_game(db: Session, game_id: int, winner: Optional[int] = None) -> Optional[Game]:
    """
    Mark game as finished.

    Args:
        db: Database session
        game_id: Game ID
        winner: Winner player ID

    Returns:
        Updated game or None
    """
    game = get_game(db, game_id)
    if game:
        game.is_finished = True
        game.finished_at = datetime.utcnow()
        game.winner = winner
        db.commit()
        db.refresh(game)
    return game


def add_action(
    db: Session,
    game_id: int,
    player_id: int,
    action_type: str,
    action_data: Optional[Dict[str, Any]] = None,
    turn_number: int = 0,
) -> GameAction:
    """
    Add action to game history.

    Args:
        db: Database session
        game_id: Game ID
        player_id: Player ID
        action_type: Action type ('bid', 'challenge', 'believe')
        action_data: Additional action data
        turn_number: Turn number

    Returns:
        Created action
    """
    action = GameAction(
        game_id=game_id,
        player_id=player_id,
        action_type=action_type,
        action_data=action_data or {},
        turn_number=turn_number,
    )
    db.add(action)
    db.commit()
    db.refresh(action)
    return action


def save_game_state(
    db: Session,
    game_id: int,
    turn_number: int,
    state_json: Dict[str, Any],
) -> GameState:
    """
    Save game state snapshot.

    Args:
        db: Database session
        game_id: Game ID
        turn_number: Turn number
        state_json: Serialized game state

    Returns:
        Created state snapshot
    """
    state = GameState(
        game_id=game_id,
        turn_number=turn_number,
        state_json=state_json,
    )
    db.add(state)
    db.commit()
    db.refresh(state)
    return state


def get_game_actions(db: Session, game_id: int) -> List[GameAction]:
    """
    Get all actions for a game.

    Args:
        db: Database session
        game_id: Game ID

    Returns:
        List of actions
    """
    return db.query(GameAction).filter(GameAction.game_id == game_id).order_by(GameAction.turn_number).all()


def get_game_history(db: Session, game_id: int) -> Dict[str, Any]:
    """
    Get game history with actions and states.

    Args:
        db: Database session
        game_id: Game ID

    Returns:
        Dictionary with game info, actions, and states
    """
    game = get_game(db, game_id)
    if not game:
        return {}

    actions = get_game_actions(db, game_id)
    states = db.query(GameState).filter(GameState.game_id == game_id).order_by(GameState.turn_number).all()

    return {
        "game": {
            "id": game.id,
            "created_at": game.created_at.isoformat() if game.created_at else None,
            "finished_at": game.finished_at.isoformat() if game.finished_at else None,
            "winner": game.winner,
            "num_players": game.num_players,
            "is_finished": game.is_finished,
        },
        "players": [
            {
                "player_id": p.player_id,
                "player_type": p.player_type,
                "model_path": p.model_path,
            }
            for p in game.players
        ],
        "actions": [
            {
                "id": a.id,
                "player_id": a.player_id,
                "action_type": a.action_type,
                "action_data": a.action_data,
                "timestamp": a.timestamp.isoformat() if a.timestamp else None,
                "turn_number": a.turn_number,
            }
            for a in actions
        ],
        "states": [
            {
                "turn_number": s.turn_number,
                "state_json": s.state_json,
                "timestamp": s.timestamp.isoformat() if s.timestamp else None,
            }
            for s in states
        ],
    }


def get_player_statistics(db: Session) -> Dict[str, Any]:
    """
    Get player statistics.

    Returns:
        Dictionary with player statistics
    """
    total_games = db.query(Game).filter(Game.is_finished == True).count()
    games_won = db.query(Game).filter(Game.is_finished == True, Game.winner == 0).count()

    winrate = games_won / total_games if total_games > 0 else 0.0

    # Get average game duration
    finished_games = db.query(Game).filter(Game.is_finished == True, Game.finished_at.isnot(None)).all()
    durations = []
    for game in finished_games:
        if game.created_at and game.finished_at:
            duration = (game.finished_at - game.created_at).total_seconds()
            durations.append(duration)

    avg_duration = sum(durations) / len(durations) if durations else 0.0

    return {
        "total_games": total_games,
        "games_won": games_won,
        "winrate": winrate,
        "avg_duration_seconds": avg_duration,
    }


def get_model_statistics(db: Session) -> Dict[str, Any]:
    """
    Get statistics by model.

    Returns:
        Dictionary with model statistics
    """
    from sqlalchemy import func

    # Count games by model
    model_stats = (
        db.query(
            GamePlayer.model_path,
            func.count(Game.id).label("games_count"),
        )
        .join(Game, GamePlayer.game_id == Game.id)
        .filter(Game.is_finished == True, GamePlayer.player_type == "ai")
        .group_by(GamePlayer.model_path)
        .all()
    )

    # Count wins by model (as opponent)
    model_wins = (
        db.query(
            GamePlayer.model_path,
            func.count(Game.id).label("wins_count"),
        )
        .join(Game, GamePlayer.game_id == Game.id)
        .filter(
            Game.is_finished == True,
            GamePlayer.player_type == "ai",
            Game.winner == GamePlayer.player_id,
        )
        .group_by(GamePlayer.model_path)
        .all()
    )

    wins_dict = {model_path: wins for model_path, wins in model_wins}

    result = {}
    for model_path, games_count in model_stats:
        result[model_path or "unknown"] = {
            "games_count": games_count,
            "wins_count": wins_dict.get(model_path, 0),
            "winrate": wins_dict.get(model_path, 0) / games_count if games_count > 0 else 0.0,
        }

    return result

