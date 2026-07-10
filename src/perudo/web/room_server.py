"""
Room management and multiplayer orchestration for web games.
"""

import asyncio
import random
import string
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from .database.database import SessionLocal
from .database.operations import create_game
from .game_server import GameSession


ROOM_SIZE = 4


@dataclass
class Seat:
    """A seat in a multiplayer room."""

    player_id: int
    seat_type: str = "empty"
    player_name: Optional[str] = None
    player_token: Optional[str] = None
    model_path: Optional[str] = None
    is_connected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "seat_type": self.seat_type,
            "player_name": self.player_name,
            "is_connected": self.is_connected,
        }


class Room:
    """Lobby and running game state for one room."""

    def __init__(self, room_id: str, join_code: str):
        self.room_id = room_id
        self.join_code = join_code
        self.status = "lobby"
        self.seats: List[Seat] = [Seat(player_id=i) for i in range(ROOM_SIZE)]
        self.game: Optional[GameSession] = None
        self.bot_task: Optional[asyncio.Task] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "room_id": self.room_id,
            "join_code": self.join_code,
            "status": self.status,
            "seats": [seat.to_dict() for seat in self.seats],
            "game_id": self.game.game_id if self.game else None,
        }

    def find_seat_by_token(self, player_token: str) -> Optional[Seat]:
        for seat in self.seats:
            if seat.player_token == player_token:
                return seat
        return None

    def first_empty_seat(self) -> Optional[Seat]:
        for seat in self.seats:
            if seat.seat_type == "empty":
                return seat
        return None


class RoomServer:
    """In-memory room registry and game coordinator."""

    def __init__(self, game_server):
        self.rooms: Dict[str, Room] = {}
        self.game_server = game_server

    def create_room(self) -> Room:
        room_id = str(uuid.uuid4())
        join_code = self._make_join_code()
        room = Room(room_id=room_id, join_code=join_code)
        self.rooms[room_id] = room
        return room

    def get_room(self, room_id: str) -> Optional[Room]:
        return self.rooms.get(room_id)

    def join_room(
        self,
        room_id: str,
        player_name: str,
        player_token: Optional[str] = None,
    ) -> Tuple[Room, Seat, str]:
        room = self._require_room(room_id)
        if room.status != "lobby":
            raise ValueError("Room is not joinable")

        token = player_token or str(uuid.uuid4())
        existing_seat = room.find_seat_by_token(token)
        if existing_seat:
            existing_seat.is_connected = True
            return room, existing_seat, token

        seat = room.first_empty_seat()
        if seat is None:
            raise ValueError("Room is full")

        seat.seat_type = "human"
        seat.player_name = player_name.strip() or f"Player {seat.player_id + 1}"
        seat.player_token = token
        seat.is_connected = True
        return room, seat, token

    def add_bot(
        self,
        room_id: str,
        model_path: str,
        player_name: Optional[str] = None,
    ) -> Tuple[Room, Seat]:
        room = self._require_room(room_id)
        if room.status != "lobby":
            raise ValueError("Cannot add bots after game start")

        seat = room.first_empty_seat()
        if seat is None:
            raise ValueError("Room is full")

        seat.seat_type = "bot"
        seat.player_name = player_name or f"Bot {seat.player_id + 1}"
        seat.model_path = model_path
        seat.is_connected = True
        return room, seat

    def remove_bot(self, room_id: str, player_id: int) -> Room:
        room = self._require_room(room_id)
        if room.status != "lobby":
            raise ValueError("Cannot remove bots after game start")
        if player_id < 0 or player_id >= len(room.seats):
            raise ValueError("Seat not found")

        seat = room.seats[player_id]
        if seat.seat_type != "bot":
            raise ValueError("Seat is not occupied by a bot")

        room.seats[player_id] = Seat(player_id=player_id)
        return room

    def start_room(self, room_id: str) -> Room:
        room = self._require_room(room_id)
        if room.status != "lobby":
            raise ValueError("Room has already started")
        if any(seat.seat_type == "empty" for seat in room.seats):
            raise ValueError("Room must have 4 occupied seats")

        players_info = []
        ai_player_model_paths: Dict[int, str] = {}
        human_player_ids: List[int] = []
        player_names: Dict[int, str] = {}

        for seat in room.seats:
            player_names[seat.player_id] = seat.player_name or f"Player {seat.player_id + 1}"
            players_info.append({
                "player_id": seat.player_id,
                "player_type": seat.seat_type,
                "model_path": seat.model_path,
                "display_name": seat.player_name,
                "seat_type": seat.seat_type,
                "join_token": seat.player_token,
            })
            if seat.seat_type == "bot":
                if not seat.model_path:
                    raise ValueError("Bot seat is missing model path")
                ai_player_model_paths[seat.player_id] = seat.model_path
            elif seat.seat_type == "human":
                human_player_ids.append(seat.player_id)

        with SessionLocal() as db:
            db_game = create_game(db, num_players=ROOM_SIZE, players_info=players_info)

        game_id = str(uuid.uuid4())
        room.game = GameSession(
            game_id=game_id,
            model_paths=list(ai_player_model_paths.values()),
            db_game_id=db_game.id,
            ai_player_model_paths=ai_player_model_paths,
            human_player_ids=human_player_ids,
            player_names=player_names,
        )
        room.status = "playing"
        self.game_server.sessions[game_id] = room.game
        return room

    async def make_action(self, room_id: str, player_token: str, action: int) -> Dict[str, Any]:
        room, seat = self._require_playing_room_and_seat(room_id, player_token)
        assert room.game is not None
        return await room.game.make_human_action(action, player_id=seat.player_id)

    async def continue_round(self, room_id: str, player_token: str) -> Dict[str, Any]:
        room, _seat = self._require_playing_room_and_seat(room_id, player_token)
        assert room.game is not None
        await room.game.continue_to_next_round()
        return {"success": True}

    async def process_bot_turns(
        self,
        room: Room,
        broadcast: Callable[[], Awaitable[None]],
    ) -> None:
        if room.game is None or room.bot_task is not None and not room.bot_task.done():
            return

        async def runner():
            assert room.game is not None
            async for _turn_result in room.game.process_ai_turns_streaming():
                await broadcast()
            if room.game.game_over:
                room.status = "finished"
            await broadcast()

        room.bot_task = asyncio.create_task(runner())

    def state_for_token(self, room: Room, player_token: str) -> Optional[Dict[str, Any]]:
        seat = room.find_seat_by_token(player_token)
        if seat is None or room.game is None:
            return None
        return room.game.get_state_for_player(seat.player_id)

    def set_connected(self, room_id: str, player_token: str, is_connected: bool) -> Optional[Room]:
        room = self.get_room(room_id)
        if room is None:
            return None
        seat = room.find_seat_by_token(player_token)
        if seat is not None:
            seat.is_connected = is_connected
        return room

    def _require_room(self, room_id: str) -> Room:
        room = self.get_room(room_id)
        if room is None:
            raise ValueError("Room not found")
        return room

    def _require_playing_room_and_seat(self, room_id: str, player_token: str) -> Tuple[Room, Seat]:
        room = self._require_room(room_id)
        if room.status not in ("playing", "finished") or room.game is None:
            raise ValueError("Room is not playing")
        seat = room.find_seat_by_token(player_token)
        if seat is None or seat.seat_type != "human":
            raise ValueError("Player is not in this room")
        return room, seat

    def _make_join_code(self) -> str:
        alphabet = string.ascii_uppercase + string.digits
        while True:
            code = "".join(random.choice(alphabet) for _ in range(6))
            if all(room.join_code != code for room in self.rooms.values()):
                return code
