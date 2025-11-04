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
    
    # Player with 5 dice cannot call pacao
    can_pacao, msg = PerudoRules.can_call_pacao(game_state, 1)
    assert not can_pacao

    # Player with 1 die can call pacao
    game_state.player_dice_count[1] = 1
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
    
    # The second player must have available actions: bid, challenge
    actions = PerudoRules.get_available_actions(game_state, 1)
    assert len(actions) > 0
    assert any(action[0] == "challenge" for action in actions)
    assert not any(action[0] == "pacao" for action in actions)
    assert any(action[0] == "bid" for action in actions)
    
    # Player with 1 die should have pacao available
    game_state.player_dice_count[1] = 1
    actions = PerudoRules.get_available_actions(game_state, 1)
    assert any(action[0] == "pacao" for action in actions)


def test_special_round_value_cannot_change():
    """Test that value cannot change in special round."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.player_dice_count[0] = 1
    game_state.roll_dice()  # This activates special round
    
    # Set first bid
    game_state.set_bid(0, 2, 4)
    game_state.current_player = 1
    
    # In special round, cannot change value
    is_valid, msg = PerudoRules.is_valid_bid(game_state, 1, 3, 5)
    assert not is_valid
    assert "special round" in msg.lower()
    
    # Can increase quantity with same value
    is_valid, msg = PerudoRules.is_valid_bid(game_state, 1, 3, 4)
    assert is_valid


def test_challenge_with_special_round():
    """Test challenge counting in special round (ones not jokers)."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.player_dice_count[0] = 1
    game_state.special_round_active = True
    
    # Set dice: player 0 has [1, 1, 3], player 1 has [2, 2, 3, 3, 5]
    game_state.player_dice = [[1, 1, 3], [2, 2, 3, 3, 5]]
    
    # Set bid: 5 threes (expecting 1s to count as jokers, but they shouldn't in special round)
    game_state.set_bid(0, 5, 3)
    game_state.current_player = 1
    
    # Challenge: actual count should be 3 (only the 3s, not the 1s)
    success, actual_count, bid_quantity = game_state.challenge_bid(1)
    assert actual_count == 3  # 1 from player 0 + 2 from player 1
    assert success  # Challenge succeeds because 3 < 5


def test_challenge_with_normal_round():
    """Test challenge counting in normal round (ones are jokers)."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.special_round_active = False
    
    # Set dice: player 0 has [1, 1, 3], player 1 has [2, 2, 3, 3, 5]
    game_state.player_dice = [[1, 1, 3], [2, 2, 3, 3, 5]]
    
    # Set bid: 5 threes (expecting 1s to count as jokers)
    game_state.set_bid(0, 5, 3)
    game_state.current_player = 1
    
    # Challenge: actual count should be 5 (3s + 1s as jokers)
    success, actual_count, bid_quantity = game_state.challenge_bid(1)
    assert actual_count == 5  # 1, 1, 3 from player 0 + 3, 3 from player 1 (1s count as 3s)
    assert not success  # Challenge fails because 5 >= 5