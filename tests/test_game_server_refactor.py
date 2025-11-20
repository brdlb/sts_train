import unittest
from unittest.mock import MagicMock, patch
import json
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.perudo.web.game_server import GameSession, PerudoJSONEncoder
from src.perudo.game.perudo_env import PerudoEnv

class TestPerudoJSONEncoder(unittest.TestCase):
    def test_encoding(self):
        encoder = PerudoJSONEncoder()
        
        # Test numpy types
        self.assertEqual(encoder.default(np.int64(42)), 42)
        self.assertEqual(encoder.default(np.float64(3.14)), 3.14)
        self.assertEqual(encoder.default(np.array([1, 2, 3])), [1, 2, 3])
        
        # Test nested structure via json.dumps
        data = {
            "a": np.int64(1),
            "b": np.array([1, 2]),
            "c": {"d": np.float64(1.5)}
        }
        json_str = json.dumps(data, cls=PerudoJSONEncoder)
        decoded = json.loads(json_str)
        
        self.assertEqual(decoded["a"], 1)
        self.assertEqual(decoded["b"], [1, 2])
        self.assertEqual(decoded["c"]["d"], 1.5)

class TestGameSession(unittest.TestCase):
    @patch('src.perudo.web.game_server.validate_environment_config')
    @patch('src.perudo.web.game_server.create_game')
    @patch('src.perudo.web.game_server.SessionLocal')
    @patch('src.perudo.web.game_server.RLAgent')
    @patch('os.path.exists')
    def setUp(self, mock_exists, mock_agent, mock_session, mock_create_game, mock_validate):
        mock_exists.return_value = True
        self.mock_session = mock_session
        
        # Mock web_config
        with patch('src.perudo.web.game_server.web_config') as mock_config:
            mock_config.dice_per_player = 5
            mock_config.total_dice_values = 6
            mock_config.max_quantity = 30
            mock_config.history_length = 10
            mock_config.transformer_history_length = 10
            mock_config.debug = False
            
            self.session = GameSession(
                game_id="test_game",
                model_paths=["m1", "m2", "m3"],
                db_game_id=1
            )

    def test_get_public_state(self):
        state = self.session.get_public_state()
        
        # Verify structure
        self.assertIn("game_id", state)
        self.assertIn("current_player", state)
        self.assertIn("player_dice", state)
        self.assertIsInstance(state["player_dice"]["dice_values"], list)
        
        # Verify serialization works (no numpy types remaining)
        json.dumps(state)  # Should not raise TypeError

    @patch('src.perudo.web.game_server.add_action')
    @patch('src.perudo.web.game_server.save_game_state')
    def test_process_action(self, mock_save, mock_add):
        # Mock env step return
        # obs, reward, terminated, truncated, info
        info = {
            "action_info": {
                "action_valid": True,
                "challenge_success": True,
                "dice_lost": 1,
                "loser_id": 1,
                "actual_count": 5
            }
        }
        
        # Manually call _process_action
        result = self.session._process_action(
            player_id=0,
            action=0, # Challenge (usually)
            reward=1.0,
            terminated=False,
            truncated=False,
            info=info
        )
        
        self.assertIn("action", result)
        self.assertIn("reward", result)
        self.assertEqual(result["reward"], 1.0)
        
        # Verify extended history
        self.assertEqual(len(self.session.extended_action_history), 1)
        entry = self.session.extended_action_history[0]
        self.assertEqual(entry["player_id"], 0)
        self.assertEqual(entry["consequences"]["dice_lost"], 1)

if __name__ == '__main__':
    unittest.main()
