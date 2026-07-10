import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.perudo.web.game_server import GameSession
from src.perudo.web.room_server import RoomServer


def test_room_join_and_rejoin_preserves_seat():
    server = RoomServer(game_server=SimpleNamespace(sessions={}))
    room = server.create_room()

    _room, seat, token = server.join_room(room.room_id, "Alice")
    assert seat.player_id == 0
    assert seat.seat_type == "human"
    assert seat.player_name == "Alice"

    _room, same_seat, same_token = server.join_room(room.room_id, "Alice Again", token)
    assert same_seat.player_id == seat.player_id
    assert same_token == token


def test_room_rejects_join_when_full():
    server = RoomServer(game_server=SimpleNamespace(sessions={}))
    room = server.create_room()

    for index in range(4):
        server.join_room(room.room_id, f"Player {index}")

    try:
        server.join_room(room.room_id, "Late Player")
    except ValueError as exc:
        assert "full" in str(exc)
    else:
        raise AssertionError("Expected full room to reject another player")


def test_room_can_remove_bot_from_lobby():
    server = RoomServer(game_server=SimpleNamespace(sessions={}))
    room = server.create_room()
    _room, bot_seat = server.add_bot(room.room_id, "model-a.zip", "Bot A")

    updated_room = server.remove_bot(room.room_id, bot_seat.player_id)

    seat = updated_room.seats[bot_seat.player_id]
    assert seat.seat_type == "empty"
    assert seat.player_name is None


@patch("src.perudo.web.room_server.GameSession")
@patch("src.perudo.web.room_server.create_game")
@patch("src.perudo.web.room_server.SessionLocal")
def test_start_room_builds_human_and_bot_mapping(mock_session_local, mock_create_game, mock_game_session):
    mock_session_local.return_value.__enter__.return_value = MagicMock()
    mock_create_game.return_value = SimpleNamespace(id=123)
    fake_game = SimpleNamespace(game_id="game-1")
    mock_game_session.return_value = fake_game

    backing_game_server = SimpleNamespace(sessions={})
    server = RoomServer(game_server=backing_game_server)
    room = server.create_room()
    server.join_room(room.room_id, "Alice")
    server.join_room(room.room_id, "Bob")
    server.add_bot(room.room_id, "model-a.zip", "Bot A")
    server.add_bot(room.room_id, "model-b.zip", "Bot B")

    started_room = server.start_room(room.room_id)

    assert started_room.status == "playing"
    assert list(backing_game_server.sessions.values()) == [fake_game]
    kwargs = mock_game_session.call_args.kwargs
    assert kwargs["human_player_ids"] == [0, 1]
    assert kwargs["ai_player_model_paths"] == {
        2: "model-a.zip",
        3: "model-b.zip",
    }


def test_game_session_filters_dice_by_player():
    with patch("src.perudo.web.game_server.validate_environment_config"), \
        patch("src.perudo.web.game_server.SessionLocal"), \
        patch("src.perudo.web.game_server.save_game_state"):
        session = GameSession(
            game_id="privacy-game",
            model_paths=[],
            db_game_id=1,
            ai_player_model_paths={},
            human_player_ids=[0, 1, 2, 3],
            player_names={0: "Alice", 1: "Bob"},
        )

    session.env.game_state.player_dice = [
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3],
        [4, 4, 4, 4, 4],
    ]

    state_for_alice = session.get_state_for_player(0)
    state_for_bob = session.get_state_for_player(1)

    assert state_for_alice["my_player_id"] == 0
    assert state_for_bob["my_player_id"] == 1
    assert state_for_alice["player_dice"]["dice_values"] == [1, 1, 1, 1, 1]
    assert state_for_bob["player_dice"]["dice_values"] == [2, 2, 2, 2, 2]


def test_wrong_human_cannot_act_out_of_turn():
    with patch("src.perudo.web.game_server.validate_environment_config"), \
        patch("src.perudo.web.game_server.SessionLocal"), \
        patch("src.perudo.web.game_server.save_game_state"):
        session = GameSession(
            game_id="turn-game",
            model_paths=[],
            db_game_id=1,
            ai_player_model_paths={},
            human_player_ids=[0, 1, 2, 3],
        )

    session.env.game_state.current_player = 0

    result = asyncio.run(session.make_human_action(action=2, player_id=1))

    assert result == {"error": "Not this player's turn"}


def test_room_server_processes_bot_turns():
    class FakeGame:
        game_id = "game-1"
        game_over = False

        async def process_ai_turns_streaming(self):
            yield {"player_id": 2}

    async def run_test():
        server = RoomServer(game_server=SimpleNamespace(sessions={}))
        room = server.create_room()
        room.game = FakeGame()
        calls = []

        async def broadcast():
            calls.append("broadcast")

        await server.process_bot_turns(room, broadcast)
        await room.bot_task
        assert calls == ["broadcast", "broadcast"]

    asyncio.run(run_test())
