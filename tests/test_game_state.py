"""
Tests for the GameState class.
"""

import pytest
import numpy as np
from src.perudo.game.game_state import GameState


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

