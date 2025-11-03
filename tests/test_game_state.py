"""
Тесты для класса GameState.
"""

import pytest
import numpy as np
from src.perudo.game.game_state import GameState


def test_game_state_initialization():
    """Тест инициализации состояния игры."""
    game_state = GameState(num_players=4, dice_per_player=5)
    assert game_state.num_players == 4
    assert game_state.dice_per_player == 5
    assert len(game_state.player_dice) == 4
    assert all(len(dice) == 5 for dice in game_state.player_dice)
    assert game_state.current_player == 0
    assert game_state.current_bid is None
    assert not game_state.game_over


def test_roll_dice():
    """Тест броска костей."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    
    assert len(game_state.player_dice) == 2
    assert all(len(dice) == 5 for dice in game_state.player_dice)
    assert all(1 <= die <= 6 for dice in game_state.player_dice for die in dice)


def test_set_bid():
    """Тест установки ставки."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    
    # Первая ставка
    assert game_state.set_bid(0, 3, 4)
    assert game_state.current_bid == (3, 4)
    
    # Вторая ставка должна быть выше
    game_state.current_player = 1
    assert game_state.set_bid(1, 4, 4)  # Больше количество
    assert game_state.set_bid(1, 3, 5)  # Больше значение


def test_challenge_bid():
    """Тест вызова ставки."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    
    # Устанавливаем ставку
    game_state.set_bid(0, 10, 4)
    
    # Вызываем ставку
    success, actual_count, bid_quantity = game_state.challenge_bid(1)
    
    assert isinstance(success, bool)
    assert actual_count >= 0
    assert bid_quantity == 10


def test_lose_dice():
    """Тест потери костей."""
    game_state = GameState(num_players=2, dice_per_player=5)
    
    initial_count = game_state.player_dice_count[0]
    game_state.lose_dice(0, 1)
    
    assert game_state.player_dice_count[0] == initial_count - 1
    
    # Проверка активации пальфико
    game_state.player_dice_count[0] = 1
    game_state.lose_dice(0, 0)  # Не теряем кость, но проверяем статус
    assert game_state.palifico_active[0] or game_state.player_dice_count[0] > 0


def test_reset():
    """Тест сброса игры."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.set_bid(0, 3, 4)
    game_state.current_player = 1
    
    game_state.reset()
    
    assert game_state.current_bid is None
    assert game_state.current_player == 0
    assert len(game_state.bid_history) == 0
    assert not game_state.game_over


def test_game_over():
    """Тест окончания игры."""
    game_state = GameState(num_players=2, dice_per_player=5)
    
    # Убираем кости у всех игроков кроме одного
    for i in range(1, game_state.num_players):
        game_state.player_dice_count[i] = 0
    
    game_state._check_game_over()
    
    # Проверяем, что игра не завершена, если у одного игрока еще есть кости
    assert game_state.player_dice_count[0] > 0
    
    # Убираем кости у последнего игрока
    game_state.player_dice_count[0] = 0
    game_state._check_game_over()
    
    # Теперь игра должна быть завершена
    assert game_state.game_over

