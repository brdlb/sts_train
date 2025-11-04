"""
Tests for Perudo rules.
"""

import pytest
from src.perudo.game.game_state import GameState
from src.perudo.game.rules import PerudoRules


def test_is_valid_bid():
    """Test bid validation."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    
    # Valid first bid
    is_valid, msg = PerudoRules.is_valid_bid(game_state, 0, 3, 4)
    assert is_valid
    
    # Invalid bid (not your turn)
    is_valid, msg = PerudoRules.is_valid_bid(game_state, 1, 3, 4)
    assert not is_valid
    
    # Set first bid
    game_state.set_bid(0, 3, 4)
    game_state.current_player = 1
    
    # Valid second bid (higher)
    is_valid, msg = PerudoRules.is_valid_bid(game_state, 1, 4, 4)
    assert is_valid
    
    # Invalid second bid (lower)
    is_valid, msg = PerudoRules.is_valid_bid(game_state, 1, 2, 4)
    assert not is_valid


def test_can_challenge():
    """Test challenge possibility."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    
    # Cannot challenge if no bid
    can_challenge, msg = PerudoRules.can_challenge(game_state, 0)
    assert not can_challenge
    
    # Set bid
    game_state.set_bid(0, 3, 4)
    game_state.current_player = 1
    
    # Now can challenge
    can_challenge, msg = PerudoRules.can_challenge(game_state, 1)
    assert can_challenge


def test_can_call_pacao():
    """Test pacao call possibility."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    
    # Cannot call pacao if no bid
    can_pacao, msg = PerudoRules.can_call_pacao(game_state, 0)
    assert not can_pacao
    
    # Set bid
    game_state.set_bid(0, 3, 4)
    game_state.current_player = 1
    
    # Now can call pacao
    can_pacao, msg = PerudoRules.can_call_pacao(game_state, 1)
    assert can_pacao
    
    # Cannot call pacao again after it's been called
    game_state.pacao_called = True
    can_pacao, msg = PerudoRules.can_call_pacao(game_state, 1)
    assert not can_pacao


def test_process_challenge_result():
    """Test processing challenge result."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    game_state.set_bid(0, 10, 4)
    
    # Simulate successful challenge
    loser_id, dice_lost = PerudoRules.process_challenge_result(
        game_state, 1, True, 5, 10
    )
    
    assert loser_id == 0  # The one who made the bid loses
    assert dice_lost == 1
    
    # Simulate failed challenge
    loser_id, dice_lost = PerudoRules.process_challenge_result(
        game_state, 1, False, 12, 10
    )
    
    assert loser_id == 1  # The challenger loses
    assert dice_lost == 1


def test_get_available_actions():
    """Test getting available actions."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    
    # The first player must have bid actions available
    actions = PerudoRules.get_available_actions(game_state, 0)
    assert len(actions) > 0
    assert any(action[0] == "bid" for action in actions)
    
    # Set a bid
    game_state.set_bid(0, 3, 4)
    game_state.current_player = 1
    
    # The second player must have available actions: bid, challenge, and pacao
    actions = PerudoRules.get_available_actions(game_state, 1)
    assert len(actions) > 0
    assert any(action[0] == "challenge" for action in actions)
    assert any(action[0] == "pacao" for action in actions)
    assert any(action[0] == "bid" for action in actions)

