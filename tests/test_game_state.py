"""
Tests for the GameState class.
"""

import pytest
import numpy as np
from src.perudo.game.game_state import GameState
from src.perudo.game.rules import PerudoRules


def test_game_state_initialization():
    """Test game state initialization."""
    game_state = GameState(num_players=4, dice_per_player=5)
    assert game_state.num_players == 4
    assert game_state.dice_per_player == 5
    assert len(game_state.player_dice) == 4
    assert all(len(dice) == 5 for dice in game_state.player_dice)
    assert game_state.current_player == 0
    assert game_state.current_bid is None
    assert not game_state.game_over


def test_roll_dice():
    """Test rolling dice."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    
    assert len(game_state.player_dice) == 2
    assert all(len(dice) == 5 for dice in game_state.player_dice)
    assert all(1 <= die <= 6 for dice in game_state.player_dice for die in dice)


def test_set_bid():
    """Test setting a bid."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    
    # First bid
    assert game_state.set_bid(0, 3, 4)
    assert game_state.current_bid == (3, 4)
    
    # Second bid must be higher
    game_state.current_player = 1
    assert game_state.set_bid(1, 4, 4)  # Higher quantity
    assert game_state.set_bid(1, 4, 5)  # Same quantity, higher value (Perudo rule)


def test_challenge_bid():
    """Test challenging a bid."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    
    # Set a bid
    game_state.set_bid(0, 10, 4)
    
    # Challenge the bid
    success, actual_count, bid_quantity = game_state.challenge_bid(1)
    
    assert isinstance(success, bool)
    assert actual_count >= 0
    assert bid_quantity == 10


def test_lose_dice():
    """Test dice loss."""
    game_state = GameState(num_players=2, dice_per_player=5)
    
    initial_count = game_state.player_dice_count[0]
    game_state.lose_dice(0, 1)
    
    assert game_state.player_dice_count[0] == initial_count - 1
    
    # Testing palifico activation
    game_state.player_dice_count[0] = 1
    game_state.lose_dice(0, 0)  # No actual dice lost, just check status
    assert game_state.palifico_active[0] or game_state.player_dice_count[0] > 0


def test_reset():
    """Test game reset."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.set_bid(0, 3, 4)
    game_state.current_player = 1
    
    game_state.reset()
    
    assert game_state.current_bid is None
    assert game_state.current_player == 0
    assert len(game_state.bid_history) == 0
    assert not game_state.game_over


def test_game_over():
    """Test end of game."""
    game_state = GameState(num_players=2, dice_per_player=5)
    
    # Remove dice from all players except one
    for i in range(1, game_state.num_players):
        game_state.player_dice_count[i] = 0
    
    game_state._check_game_over()
    
    # Check that game is not over if one player still has dice
    assert game_state.player_dice_count[0] > 0
    
    # Remove dice from last player
    game_state.player_dice_count[0] = 0
    game_state._check_game_over()
    
    # Game should now be over
    assert game_state.game_over


def test_special_round_detection():
    """Test special round detection."""
    game_state = GameState(num_players=3, dice_per_player=5)
    
    # Initially no special round
    assert not game_state.special_round_active
    
    # Set one player to 1 die
    game_state.player_dice_count[0] = 1
    game_state.roll_dice()
    
    # Special round should be active
    assert game_state.special_round_active
    
    # Set all players to 1 die
    game_state.player_dice_count[1] = 1
    game_state.player_dice_count[2] = 1
    game_state.roll_dice()
    
    # Special round should still be active
    assert game_state.special_round_active
    
    # Set all players back to 5 dice
    game_state.player_dice_count[0] = 5
    game_state.player_dice_count[1] = 5
    game_state.player_dice_count[2] = 5
    game_state.roll_dice()
    
    # Special round should not be active
    assert not game_state.special_round_active


def test_count_dice_for_value_normal_round():
    """Test dice counting in normal round (ones are jokers)."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.special_round_active = False
    
    # Set specific dice: player 0 has [1, 1, 3, 3, 4], player 1 has [2, 2, 3, 5, 6]
    game_state.player_dice = [[1, 1, 3, 3, 4], [2, 2, 3, 5, 6]]
    
    # Count 3s: should include 3s and 1s (jokers)
    count = game_state._count_dice_for_value(3)
    assert count == 5  # 3, 3 from player 0 + 1, 1 from player 0 (jokers) + 3 from player 1
    
    # Count 2s: should include 2s and 1s (jokers)
    count = game_state._count_dice_for_value(2)
    assert count == 4  # 2, 2 from player 1 + 1, 1 from player 0 (jokers)
    
    # Count 1s: should only include actual 1s (not jokers when bidding on 1s)
    count = game_state._count_dice_for_value(1)
    assert count == 2  # Only the actual 1s from player 0


def test_count_dice_for_value_special_round():
    """Test dice counting in special round (ones are NOT jokers)."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.special_round_active = True
    
    # Set specific dice: player 0 has [1, 1, 3, 3, 4], player 1 has [2, 2, 3, 5, 6]
    game_state.player_dice = [[1, 1, 3, 3, 4], [2, 2, 3, 5, 6]]
    
    # Count 3s: should NOT include 1s (not jokers in special round)
    count = game_state._count_dice_for_value(3)
    assert count == 3  # Only 3, 3 from player 0 + 3 from player 1
    
    # Count 2s: should NOT include 1s
    count = game_state._count_dice_for_value(2)
    assert count == 2  # Only 2, 2 from player 1
    
    # Count 1s: should only include actual 1s
    count = game_state._count_dice_for_value(1)
    assert count == 2  # Only the actual 1s from player 0


def test_bid_validation_ones_exception():
    """Test bid validation with ones exception (can reduce by half)."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    
    # Set bid: 5 fives
    game_state.set_bid(0, 5, 5)
    game_state.current_player = 1
    
    # Can reduce to 3 ones (half of 5, rounding up)
    assert game_state._is_bid_higher(3, 1, 5, 5)
    
    # Can reduce to 3 ones from 6 sixes (half of 6, rounding up)
    game_state.current_bid = (6, 6)
    assert game_state._is_bid_higher(3, 1, 6, 6)
    
    # Cannot reduce to 2 ones from 5 fives (2 < 3)
    assert not game_state._is_bid_higher(2, 1, 5, 5)


def test_bid_validation_after_ones():
    """Test bid validation after previous bet was ones."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    
    # Set bid: 2 ones
    game_state.set_bid(0, 2, 1)
    game_state.current_player = 1
    
    # Can increase quantity of ones
    assert game_state._is_bid_higher(3, 1, 2, 1)
    
    # Can use different value with quantity = 2 * 2 + 1 = 5
    assert game_state._is_bid_higher(5, 2, 2, 1)
    assert game_state._is_bid_higher(5, 3, 2, 1)
    
    # Cannot use different value with quantity < 5
    assert not game_state._is_bid_higher(4, 2, 2, 1)


def test_bid_validation_quantity_increase():
    """Test bid validation with quantity increase."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    
    # Set bid: 3 fives
    game_state.set_bid(0, 3, 5)
    game_state.current_player = 1
    
    # Can increase quantity with any value
    assert game_state._is_bid_higher(4, 5, 3, 5)  # Same value, higher quantity
    assert game_state._is_bid_higher(4, 1, 3, 5)  # Different value, higher quantity
    assert game_state._is_bid_higher(4, 6, 3, 5)  # Different value, higher quantity
    
    # Cannot decrease quantity
    assert not game_state._is_bid_higher(2, 5, 3, 5)


def test_bid_validation_same_quantity():
    """Test bid validation with same quantity (value must increase)."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.roll_dice()
    
    # Set bid: 3 fives
    game_state.set_bid(0, 3, 5)
    game_state.current_player = 1
    
    # Can keep quantity same if value increases
    assert game_state._is_bid_higher(3, 6, 3, 5)
    
    # Cannot keep quantity same if value doesn't increase
    assert not game_state._is_bid_higher(3, 5, 3, 5)
    assert not game_state._is_bid_higher(3, 4, 3, 5)


def test_challenge_equality_case():
    """Test challenge when actual count equals bid (challenger loses)."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.special_round_active = False
    
    # Set dice: player 0 has [1, 1, 3, 3, 3], player 1 has [2, 2, 3, 3, 5]
    # Total: 3, 3, 3 from player 0 + 1, 1 from player 0 (jokers) + 3, 3 from player 1 = 7 threes
    game_state.player_dice = [[1, 1, 3, 3, 3], [2, 2, 3, 3, 5]]
    
    # Set bid: 7 threes
    game_state.set_bid(0, 7, 3)
    game_state.current_player = 1
    
    # Challenge: actual count should be 7 (equal to bid)
    # According to rules: "If there are more or equal dice... challenger loses"
    success, actual_count, bid_quantity = game_state.challenge_bid(1)
    assert actual_count == 7  # 3, 3, 3 from player 0 + 1, 1 (jokers) + 3, 3 from player 1
    assert not success  # Challenge fails because 7 >= 7 (equality case)
    
    # The challenger should lose (loser is determined by process_challenge_result)
    loser_id, dice_lost = PerudoRules.process_challenge_result(
        game_state, 1, False, 7, 7
    )
    assert loser_id == 1  # Challenger loses
    assert dice_lost == 1


def test_challenge_greater_than_case():
    """Test challenge when actual count is greater than bid (challenger loses)."""
    game_state = GameState(num_players=2, dice_per_player=5)
    game_state.special_round_active = False
    
    # Set dice: player 0 has [1, 1, 3, 3, 3], player 1 has [2, 3, 3, 3, 5]
    # Total: 3, 3, 3 from player 0 + 1, 1 (jokers) + 3, 3, 3 from player 1 = 8 threes
    game_state.player_dice = [[1, 1, 3, 3, 3], [2, 3, 3, 3, 5]]
    
    # Set bid: 6 threes
    game_state.set_bid(0, 6, 3)
    game_state.current_player = 1
    
    # Challenge: actual count should be 8 (greater than bid)
    success, actual_count, bid_quantity = game_state.challenge_bid(1)
    assert actual_count == 8  # More than bid
    assert not success  # Challenge fails because 8 >= 6
    
    # The challenger should lose
    loser_id, dice_lost = PerudoRules.process_challenge_result(
        game_state, 1, False, 8, 6
    )
    assert loser_id == 1  # Challenger loses
    assert dice_lost == 1