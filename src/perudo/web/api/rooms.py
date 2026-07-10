"""
API and WebSocket endpoints for multiplayer rooms.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ..room_server import Room, RoomServer


router = APIRouter(prefix="/api/rooms", tags=["rooms"])
ws_router = APIRouter(tags=["rooms"])

room_server: Optional[RoomServer] = None
connections: Dict[str, List[Dict[str, Any]]] = {}


def set_room_server(server: RoomServer):
    """Set room server instance."""
    global room_server
    room_server = server


class CreateRoomResponse(BaseModel):
    room: Dict[str, Any]


class JoinRoomRequest(BaseModel):
    player_name: str
    player_token: Optional[str] = None


class JoinRoomResponse(BaseModel):
    room: Dict[str, Any]
    player_id: int
    player_token: str


class AddBotRequest(BaseModel):
    model_path: str
    player_name: Optional[str] = None


def require_room_server() -> RoomServer:
    if room_server is None:
        raise HTTPException(status_code=500, detail="Room server not initialized")
    return room_server


@router.post("")
async def create_room():
    server = require_room_server()
    room = server.create_room()
    return {"room": room.to_dict()}


@router.get("/{room_id}")
async def get_room(room_id: str):
    server = require_room_server()
    room = server.get_room(room_id)
    if room is None:
        raise HTTPException(status_code=404, detail="Room not found")
    return {"room": room.to_dict()}


@router.post("/{room_id}/join")
async def join_room(room_id: str, request: JoinRoomRequest):
    server = require_room_server()
    try:
        room, seat, token = server.join_room(
            room_id=room_id,
            player_name=request.player_name,
            player_token=request.player_token,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await broadcast_room(room, "room_updated")
    return {
        "room": room.to_dict(),
        "player_id": seat.player_id,
        "player_token": token,
    }


@router.post("/{room_id}/bots")
async def add_bot(room_id: str, request: AddBotRequest):
    server = require_room_server()
    try:
        room, seat = server.add_bot(
            room_id=room_id,
            model_path=request.model_path,
            player_name=request.player_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await broadcast_room(room, "room_updated")
    return {"room": room.to_dict(), "player_id": seat.player_id}


@router.delete("/{room_id}/bots/{player_id}")
async def remove_bot(room_id: str, player_id: int):
    server = require_room_server()
    try:
        room = server.remove_bot(room_id, player_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await broadcast_room(room, "room_updated")
    return {"room": room.to_dict()}


@router.post("/{room_id}/start")
async def start_room(room_id: str):
    server = require_room_server()
    try:
        room = server.start_room(room_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    await broadcast_room(room, "game_started")
    await server.process_bot_turns(room, lambda: broadcast_room(room, "state_updated"))
    return {"room": room.to_dict()}


@ws_router.websocket("/ws/rooms/{room_id}")
async def room_websocket(websocket: WebSocket, room_id: str, player_token: str):
    server = require_room_server()
    room = server.get_room(room_id)
    if room is None:
        await websocket.close(code=4404)
        return
    if room.find_seat_by_token(player_token) is None:
        await websocket.close(code=4403)
        return

    await websocket.accept()
    connections.setdefault(room_id, []).append({
        "websocket": websocket,
        "player_token": player_token,
    })
    server.set_connected(room_id, player_token, True)
    await broadcast_room(room, "room_updated")

    try:
        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")

            if message_type == "ping":
                await websocket.send_json({"type": "pong", "room": room.to_dict()})
                continue

            if message_type == "make_action":
                result = await server.make_action(room_id, player_token, int(message.get("action")))
                if "error" in result:
                    await websocket.send_json({
                        "type": "action_rejected",
                        "room": room.to_dict(),
                        "error": result["error"],
                    })
                    continue

                if room.game and room.game.game_over:
                    room.status = "finished"
                    await broadcast_room(room, "game_finished")
                else:
                    await broadcast_room(room, "state_updated")
                    await server.process_bot_turns(room, lambda: broadcast_room(room, "state_updated"))
                continue

            if message_type == "continue_round":
                try:
                    await server.continue_round(room_id, player_token)
                except (RuntimeError, ValueError) as exc:
                    await websocket.send_json({
                        "type": "action_rejected",
                        "room": room.to_dict(),
                        "error": str(exc),
                    })
                    continue
                await broadcast_room(room, "state_updated")
                await server.process_bot_turns(room, lambda: broadcast_room(room, "state_updated"))
                continue

            await websocket.send_json({
                "type": "action_rejected",
                "room": room.to_dict(),
                "error": "Unknown message type",
            })
    except WebSocketDisconnect:
        pass
    finally:
        room_connections = connections.get(room_id, [])
        connections[room_id] = [
            item for item in room_connections if item["websocket"] is not websocket
        ]
        server.set_connected(room_id, player_token, False)
        await broadcast_room(room, "room_updated")


async def broadcast_room(room: Room, event_type: str) -> None:
    """Broadcast room state, with per-player game state when available."""
    active_connections = connections.get(room.room_id, [])
    stale_connections = []
    for item in active_connections:
        websocket: WebSocket = item["websocket"]
        player_token: str = item["player_token"]
        payload: Dict[str, Any] = {
            "type": event_type,
            "room": room.to_dict(),
        }
        if room.game is not None:
            payload["state"] = room.game.get_state_for_player(
                room.find_seat_by_token(player_token).player_id
            )
        try:
            await websocket.send_json(payload)
        except RuntimeError:
            stale_connections.append(item)

    if stale_connections:
        connections[room.room_id] = [
            item for item in active_connections if item not in stale_connections
        ]
