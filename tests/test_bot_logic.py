"""
Tests for bot logic functions.
"""

import unittest
import numpy as np
from src.perudo.agents.bot_logic.utils import (
    get_game_stage,
    calculate_expected_count,
    get_hand_strength,
    generate_possible_next_bids,
)
from src.perudo.agents.bot_logic.genesis import (
    should_stan_start_special_round,
    get_standard_stan_decision,
)
from src.perudo.agents.bot_logic.personalities import (
    should_others_start_special_round,
    get_personality_decision,
)
from src.perudo.agents.bot_logic.constants import BOT_PERSONALITIES
from src.perudo.game.game_state import GameState
from src.perudo.agents.bot_logic.player_analysis import create_initial_player_analysis


class TestBotLogicUtils(unittest.TestCase):
    """Test cases for bot logic utility functions."""

    def test_get_game_stage(self):
        """Test game stage determination."""
        # DUEL
        self.assertEqual(get_game_stage(10, 2), "DUEL")
        
        # CHAOS
        self.assertEqual(get_game_stage(30, 4), "CHAOS")
        
        # POSITIVE
        self.assertEqual(get_game_stage(15, 4), "POSITIVE")
        
        # TENSE
        self.assertEqual(get_game_stage(8, 4), "TENSE")
        
        # KNIFE_FIGHT
        self.assertEqual(get_game_stage(4, 3), "KNIFE_FIGHT")

    def test_calculate_expected_count(self):
        """Test expected count calculation."""
        bot_dice = [1, 2, 3, 4, 5]
        total_dice = 20
        is_special_round = False
        
        # Test normal round (1s are wild)
        expected = calculate_expected_count(2, bot_dice, total_dice, is_special_round)
        # Bot has 1 die with value 2, plus 1 wild (1), so 2 in hand
        # Others: 15 dice, probability 1/3 = 5
        # Total should be around 7
        self.assertGreater(expected, 5)
        self.assertLess(expected, 10)
        
        # Test special round (1s are not wild)
        expected_special = calculate_expected_count(2, bot_dice, total_dice, True)
        # Bot has 1 die with value 2, no wilds
        # Others: 15 dice, probability 1/6 = 2.5
        # Total should be around 3.5
        self.assertGreater(expected_special, 2)
        self.assertLess(expected_special, 5)

    def test_get_hand_strength(self):
        """Test hand strength calculation."""
        bot_dice = [1, 1, 2, 3, 4]
        is_special_round = False
        
        strength = get_hand_strength(bot_dice, is_special_round)
        
        # Should have counts for each face
        self.assertEqual(strength[1], 2)  # Two 1s
        self.assertEqual(strength[2], 3)  # One 2 + two wilds
        self.assertEqual(strength[3], 3)  # One 3 + two wilds
        self.assertEqual(strength[4], 3)  # One 4 + two wilds
        self.assertEqual(strength[5], 2)  # Zero 5s + two wilds
        self.assertEqual(strength[6], 2)  # Zero 6s + two wilds

    def test_generate_possible_next_bids(self):
        """Test possible next bids generation."""
        current_bid = (3, 2)  # 3x2
        total_dice = 20
        is_special_round = False
        
        possible_bids = generate_possible_next_bids(current_bid, total_dice, is_special_round)
        
        # Should have multiple options
        self.assertGreater(len(possible_bids), 0)
        
        # All bids should be valid
        for quantity, face in possible_bids:
            self.assertGreater(quantity, 0)
            self.assertLessEqual(quantity, total_dice)
            self.assertGreaterEqual(face, 1)
            self.assertLessEqual(face, 6)


class TestBotLogicGenesis(unittest.TestCase):
    """Test cases for Standard Stan bot logic."""

    def test_should_stan_start_special_round(self):
        """Test Standard Stan special round decision."""
        bot_dice = [1]  # One die
        
        # With few dice, should be more likely to start
        result = should_stan_start_special_round(bot_dice, 5)
        self.assertIsInstance(result, bool)
        
        # With many dice, should be less likely
        result2 = should_stan_start_special_round(bot_dice, 25)
        self.assertIsInstance(result2, bool)

    def test_get_standard_stan_decision(self):
        """Test Standard Stan decision making."""
        game_state = GameState(num_players=4, dice_per_player=5)
        bot_id = 0
        bot_dice = game_state.get_player_dice(bot_id)
        player_analysis = {}
        for player_id in range(4):
            player_analysis[player_id] = create_initial_player_analysis(player_id)
        round_bid_history = []
        
        # Test initial bid
        decision, bid = get_standard_stan_decision(
            game_state, bot_id, bot_dice, player_analysis, round_bid_history
        )
        
        self.assertEqual(decision, "BID")
        self.assertIsNotNone(bid)
        quantity, face = bid
        self.assertGreater(quantity, 0)
        self.assertGreaterEqual(face, 1)
        self.assertLessEqual(face, 6)


class TestBotLogicPersonalities(unittest.TestCase):
    """Test cases for personality-based bot logic."""

    def test_should_others_start_special_round(self):
        """Test other bots special round decision."""
        personality_name = BOT_PERSONALITIES["AGGRESSIVE"].name
        
        result = should_others_start_special_round(personality_name, 10)
        self.assertIsInstance(result, bool)

    def test_get_personality_decision(self):
        """Test personality-based decision making."""
        game_state = GameState(num_players=4, dice_per_player=5)
        bot_id = 1
        bot_dice = game_state.get_player_dice(bot_id)
        personality = BOT_PERSONALITIES["CAUTIOUS"]
        player_analysis = {}
        for player_id in range(4):
            player_analysis[player_id] = create_initial_player_analysis(player_id)
        round_bid_history = []
        
        # Test initial bid
        decision, bid = get_personality_decision(
            game_state, bot_id, bot_dice, personality, player_analysis, round_bid_history
        )
        
        self.assertEqual(decision, "BID")
        self.assertIsNotNone(bid)
        quantity, face = bid
        self.assertGreater(quantity, 0)
        self.assertGreaterEqual(face, 1)
        self.assertLessEqual(face, 6)


if __name__ == "__main__":
    unittest.main()

