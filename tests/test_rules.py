"""
Тесты для правил Perudo.
"""

import pytest
from src.perudo.game.game_state import GameState
from src.perudo.game.rules import PerudoRules


def test_is_valid_bid():
    """Тест валидации ставки."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    
    # Валидная первая ставка
    is_valid, msg = PerudoRules.is_valid_bid(game_state, 0, 3, 4)
    assert is_valid
    
    # Невалидная ставка (не ваш ход)
    is_valid, msg = PerudoRules.is_valid_bid(game_state, 1, 3, 4)
    assert not is_valid
    
    # Устанавливаем первую ставку
    game_state.set_bid(0, 3, 4)
    game_state.current_player = 1
    
    # Валидная вторая ставка (больше)
    is_valid, msg = PerudoRules.is_valid_bid(game_state, 1, 4, 4)
    assert is_valid
    
    # Невалидная вторая ставка (меньше)
    is_valid, msg = PerudoRules.is_valid_bid(game_state, 1, 2, 4)
    assert not is_valid


def test_can_challenge():
    """Тест проверки возможности вызова."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    
    # Нельзя вызвать, если нет ставки
    can_challenge, msg = PerudoRules.can_challenge(game_state, 0)
    assert not can_challenge
    
    # Устанавливаем ставку
    game_state.set_bid(0, 3, 4)
    game_state.current_player = 1
    
    # Теперь можно вызвать
    can_challenge, msg = PerudoRules.can_challenge(game_state, 1)
    assert can_challenge


def test_can_call_pacao():
    """Тест проверки возможности вызова пакао."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    
    # Нельзя вызвать пакао, если нет ставки
    can_pacao, msg = PerudoRules.can_call_pacao(game_state, 0)
    assert not can_pacao
    
    # Устанавливаем ставку
    game_state.set_bid(0, 3, 4)
    game_state.current_player = 1
    
    # Теперь можно вызвать пакао
    can_pacao, msg = PerudoRules.can_call_pacao(game_state, 1)
    assert can_pacao
    
    # После вызова пакао нельзя вызвать снова
    game_state.pacao_called = True
    can_pacao, msg = PerudoRules.can_call_pacao(game_state, 1)
    assert not can_pacao


def test_process_challenge_result():
    """Тест обработки результата вызова."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    game_state.set_bid(0, 10, 4)
    
    # Симулируем успешный вызов
    loser_id, dice_lost = PerudoRules.process_challenge_result(
        game_state, 1, True, 5, 10
    )
    
    assert loser_id == 0  # Тот, кто сделал ставку, проиграл
    assert dice_lost == 1
    
    # Симулируем неуспешный вызов
    loser_id, dice_lost = PerudoRules.process_challenge_result(
        game_state, 1, False, 12, 10
    )
    
    assert loser_id == 1  # Вызывающий проиграл
    assert dice_lost == 1


def test_get_available_actions():
    """Тест получения доступных действий."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    
    # Для первого игрока должны быть доступны ставки
    actions = PerudoRules.get_available_actions(game_state, 0)
    assert len(actions) > 0
    assert any(action[0] == "bid" for action in actions)
    
    # Устанавливаем ставку
    game_state.set_bid(0, 3, 4)
    game_state.current_player = 1
    
    # Для второго игрока должны быть доступны ставки, вызов и пакао
    actions = PerudoRules.get_available_actions(game_state, 1)
    assert len(actions) > 0
    assert any(action[0] == "challenge" for action in actions)
    assert any(action[0] == "pacao" for action in actions)
    assert any(action[0] == "bid" for action in actions)

