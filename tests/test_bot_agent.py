"""
Tests for bot agents.
"""

import unittest
import numpy as np
from src.perudo.agents.bot_agent import BotAgent
from src.perudo.agents.bot_logic.constants import BOT_PERSONALITIES
from src.perudo.game.perudo_env import PerudoEnv
from src.perudo.agents.bot_logic.player_analysis import create_initial_player_analysis


class TestBotAgent(unittest.TestCase):
    """Test cases for BotAgent."""

    def setUp(self):
        """Set up test fixtures."""
        self.env = PerudoEnv(
            num_players=4,
            dice_per_player=5,
            total_dice_values=6,
            max_quantity=30,
        )
        self.personality = BOT_PERSONALITIES["STANDARD_STAN"]
        self.player_analysis = {}
        for player_id in range(4):
            self.player_analysis[player_id] = create_initial_player_analysis(player_id)

    def test_bot_agent_initialization(self):
        """Test bot agent initialization."""
        bot = BotAgent(
            agent_id=1,
            personality=self.personality,
            env=self.env,
            player_analysis=self.player_analysis,
        )
        self.assertEqual(bot.agent_id, 1)
        self.assertEqual(bot.personality.name, "Ровный Стэн")
        self.assertEqual(bot.max_quantity, 30)

    def test_bot_agent_act_initial_bid(self):
        """Test bot agent makes initial bid."""
        bot = BotAgent(
            agent_id=0,
            personality=self.personality,
            env=self.env,
            player_analysis=self.player_analysis,
        )
        
        # Reset environment to get initial observation
        obs, _ = self.env.reset()
        
        # Bot should make a bid (not challenge or believe on first turn)
        action = bot.act(obs, deterministic=False)
        
        # Action should be a bid (>= 2) or challenge (0) or believe (1)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 2 + 30 * 6)  # Max action space size

    def test_bot_agent_act_with_current_bid(self):
        """Test bot agent acts when there's a current bid."""
        bot = BotAgent(
            agent_id=1,
            personality=self.personality,
            env=self.env,
            player_analysis=self.player_analysis,
        )
        
        # Reset environment
        obs, _ = self.env.reset()
        
        # Make a bid first (player 0)
        self.env.set_active_player(0)
        first_action = bot.act(obs, deterministic=False)
        if first_action >= 2:  # It's a bid
            obs, _, _, _, _ = self.env.step(first_action)
        
        # Now bot should be able to act
        obs = self.env.get_observation_for_player(1)
        self.env.set_active_player(1)
        action = bot.act(obs, deterministic=False)
        
        # Action should be valid
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 2 + 30 * 6)

    def test_bot_agent_different_personalities(self):
        """Test bot agents with different personalities."""
        personalities_to_test = [
            "CAUTIOUS",
            "AGGRESSIVE",
            "CALCULATING",
            "STANDARD_STAN",
        ]
        
        for personality_name in personalities_to_test:
            personality = BOT_PERSONALITIES[personality_name]
            bot = BotAgent(
                agent_id=1,
                personality=personality,
                env=self.env,
                player_analysis=self.player_analysis,
            )
            
            obs, _ = self.env.reset()
            action = bot.act(obs, deterministic=False)
            
            # Action should be valid
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, 2 + 30 * 6, 
                          f"Personality {personality_name} produced invalid action {action}")

    def test_bot_should_start_special_round(self):
        """Test bot decision to start special round."""
        bot = BotAgent(
            agent_id=0,
            personality=self.personality,
            env=self.env,
            player_analysis=self.player_analysis,
        )
        
        # Set bot to have 1 die
        self.env.game_state.player_dice_count[0] = 1
        self.env.game_state.roll_dice()
        
        # Check if bot wants to start special round
        should_start = bot.should_start_special_round(self.env.game_state)
        
        # Should return boolean
        self.assertIsInstance(should_start, bool)


if __name__ == "__main__":
    unittest.main()

