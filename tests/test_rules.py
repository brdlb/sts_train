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
    """Test pacao (believe) call possibility - any player can believe."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    
    # Cannot call pacao if no bid
    can_pacao, msg = PerudoRules.can_call_pacao(game_state, 0)
    assert not can_pacao
    
    # Set bid
    game_state.set_bid(0, 3, 4)
    game_state.current_player = 1
    
    # Any player can call pacao (believe) if there's a bid
    can_pacao, msg = PerudoRules.can_call_pacao(game_state, 1)
    assert can_pacao
    
    # Player with 5 dice can also call pacao
    game_state.player_dice_count[1] = 5
    can_pacao, msg = PerudoRules.can_call_pacao(game_state, 1)
    assert can_pacao


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
    
    # The second player must have available actions: bid, challenge, believe
    actions = PerudoRules.get_available_actions(game_state, 1)
    assert len(actions) > 0
    assert any(action[0] == "challenge" for action in actions)
    assert any(action[0] == "pacao" for action in actions)  # Any player can believe
    assert any(action[0] == "bid" for action in actions)


def test_special_round_value_cannot_change():
    """Test that value cannot change in special round."""
    game_state = GameState(num_players=3, dice_per_player=5)
    game_state.player_dice_count[0] = 1
    game_state.declare_special_round(0)  # Declare special round
    game_state.roll_dice()
    
    # Set first bid (must be quantity 1 in special round)
    game_state.set_bid(0, 1, 4)
    game_state.current_player = 1
    
    # In special round, cannot change value
    is_valid, msg = PerudoRules.is_valid_bid(game_state, 1, 2, 5)
    assert not is_valid
    assert "special round" in msg.lower()
    
    # Can increase quantity with same value
    is_valid, msg = PerudoRules.is_valid_bid(game_state, 1, 2, 4)
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


def test_first_bid_cannot_be_value_1():
    """Test that first bid cannot have value 1."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    
    # First bid with value 1 should be invalid
    is_valid, msg = PerudoRules.is_valid_bid(game_state, 0, 3, 1)
    assert not is_valid
    assert "value 1" in msg.lower() or "jokers" in msg.lower()
    
    # First bid with value 2 should be valid
    is_valid, msg = PerudoRules.is_valid_bid(game_state, 0, 3, 2)
    assert is_valid


def test_believe_exact_match_gains_die():
    """Test that believer gains die if dice exactly equals bid and has < 5 dice."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.player_dice_count[0] = 3  # Believer has 3 dice
    game_state.player_dice_count[1] = 4  # Bid maker has 4 dice
    game_state.roll_dice()
    
    game_state.set_bid(0, 5, 3)
    game_state.current_player = 1
    
    # Set dice so count exactly equals bid (5 threes)
    # Player 0: [2, 3, 3] = 2 threes (no 1s)
    # Player 1: [1, 3, 3, 3] = 1 (joker) + 3 threes = 4 threes total
    # But we need exactly 5, so:
    # Player 0: [3, 3, 3] = 3 threes
    # Player 1: [1, 3, 3] = 1 (joker) + 2 threes = 3 threes total = 6, too much
    # Let's do:
    # Player 0: [2, 3, 3] = 2 threes
    # Player 1: [1, 1, 3, 3] = 2 (jokers) + 2 threes = 4 threes total = 6, still too much
    # Let's do:
    # Player 0: [2, 2, 3] = 1 three
    # Player 1: [1, 1, 3, 3] = 2 (jokers) + 2 threes = 4 threes total = 5 ✓
    game_state.player_dice = [[2, 2, 3], [1, 1, 3, 3]]
    
    pacao_success, actual_count = game_state.call_pacao(1)
    assert pacao_success  # Exact match
    assert actual_count == 5
    
    # Process result
    loser_id, dice_lost, next_round_starter = PerudoRules.process_pacao_result(
        game_state, 1, pacao_success, actual_count, 5
    )
    assert loser_id is None  # No one loses
    assert dice_lost == 0
    assert next_round_starter is None  # Believer gains die, doesn't start round


def test_believe_exact_match_starts_round():
    """Test that believer with 5 dice starts next round if dice exactly equals bid."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.player_dice_count[0] = 5  # Bid maker has 5 dice
    game_state.player_dice_count[1] = 5  # Believer has 5 dice
    game_state.roll_dice()
    
    game_state.set_bid(0, 5, 3)
    game_state.current_player = 1
    
    # Set dice so count exactly equals bid (5 threes)
    # Player 0: [2, 2, 2, 2, 2] = 0 threes
    # Player 1: [1, 1, 3, 3, 3] = 2 (jokers) + 3 threes = 5 threes total = 5 ✓
    game_state.player_dice = [[2, 2, 2, 2, 2], [1, 1, 3, 3, 3]]
    
    pacao_success, actual_count = game_state.call_pacao(1)
    assert pacao_success  # Exact match
    assert actual_count == 5
    
    # Process result
    loser_id, dice_lost, next_round_starter = PerudoRules.process_pacao_result(
        game_state, 1, pacao_success, actual_count, 5
    )
    assert loser_id is None
    assert dice_lost == 0
    assert next_round_starter == 1  # Believer starts next round


def test_believe_not_exact_match_loses_die():
    """Test that believer loses die if dice doesn't exactly equal bid."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    
    game_state.set_bid(0, 5, 3)
    game_state.current_player = 1
    
    # Set dice so count doesn't equal bid (more than bid)
    # Player 0: [1, 3, 3, 3, 3] = 1 (joker) + 4 threes = 5 threes
    # Player 1: [1, 3, 3, 3, 5] = 1 (joker) + 3 threes = 4 threes
    # Total = 9 threes (more than 5)
    game_state.player_dice = [[1, 3, 3, 3, 3], [1, 3, 3, 3, 5]]
    
    pacao_success, actual_count = game_state.call_pacao(1)
    assert not pacao_success  # Not exact match
    assert actual_count == 9  # 5 from player 0 + 4 from player 1
    
    # Process result
    loser_id, dice_lost, next_round_starter = PerudoRules.process_pacao_result(
        game_state, 1, pacao_success, actual_count, 5
    )
    assert loser_id == 1  # Believer loses
    assert dice_lost == 1
    assert next_round_starter is None


def test_special_round_declaration():
    """Test special round declaration rules."""
    game_state = GameState(num_players=4, dice_per_player=5)
    game_state.roll_dice()
    
    # Player with 1 die can declare special round if > 2 active players
    game_state.player_dice_count[0] = 1
    game_state.player_dice_count[1] = 3
    game_state.player_dice_count[2] = 2
    game_state.player_dice_count[3] = 4
    
    # Can declare special round
    assert game_state.declare_special_round(0)
    assert game_state.special_round_active
    assert game_state.special_round_declared_by == 0
    
    # Cannot declare again (already used)
    assert not game_state.declare_special_round(0)
    
    # Player with 2 dice cannot declare
    game_state.player_dice_count[1] = 2
    assert not game_state.declare_special_round(1)
    
    # With only 2 active players, cannot declare
    game_state.player_dice_count[0] = 1
    game_state.player_dice_count[2] = 0
    game_state.player_dice_count[3] = 0
    game_state.special_round_active = False
    game_state.special_round_used[0] = False
    assert not game_state.declare_special_round(0)


def test_special_round_first_bid():
    """Test first bid in special round must be quantity 1."""
    game_state = GameState(num_players=3, dice_per_player=5)
    game_state.player_dice_count[0] = 1
    game_state.declare_special_round(0)
    game_state.roll_dice()
    
    # First bid in special round must be quantity 1
    is_valid, msg = PerudoRules.is_valid_bid(game_state, 0, 1, 3)
    assert is_valid
    
    # First bid with quantity > 1 is invalid
    is_valid, msg = PerudoRules.is_valid_bid(game_state, 0, 2, 3)
    assert not is_valid
    assert "quantity 1" in msg.lower()
    
    # First bid with value 1 is allowed in special round
    is_valid, msg = PerudoRules.is_valid_bid(game_state, 0, 1, 1)
    assert is_valid


def test_get_available_actions_believe():
    """Test that believe (pacao) is available to any player after bid."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    
    # Set a bid
    game_state.set_bid(0, 3, 4)
    game_state.current_player = 1
    
    # Any player should have believe available
    actions = PerudoRules.get_available_actions(game_state, 1)
    assert any(action[0] == "pacao" for action in actions)
    
    # Player with 5 dice also has believe available
    game_state.player_dice_count[1] = 5
    actions = PerudoRules.get_available_actions(game_state, 1)
    assert any(action[0] == "pacao" for action in actions)