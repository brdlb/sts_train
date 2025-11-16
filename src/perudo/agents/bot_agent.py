"""
Bot agent implementation for Perudo game.
"""

from typing import Dict, Optional, Union
import numpy as np
from .base_agent import BaseAgent
from ..game.perudo_env import PerudoEnv
from ..game.game_state import GameState
from ..utils.helpers import bid_to_action
from .bot_logic.constants import BOT_PERSONALITIES, BotPersonality
from .bot_logic.genesis import (
    should_stan_start_special_round,
    get_standard_stan_decision,
)
from .bot_logic.personalities import (
    should_others_start_special_round,
    get_personality_decision,
)
from .bot_logic.player_analysis import (
    PlayerAnalysis,
    create_initial_player_analysis,
)


class BotAgent(BaseAgent):
    """Bot agent that uses rule-based logic to play Perudo."""

    def __init__(
        self,
        agent_id: int,
        personality: BotPersonality,
        env: PerudoEnv,
        player_analysis: Optional[Dict[int, PlayerAnalysis]] = None,
    ):
        """Initialize bot agent."""
        super().__init__(agent_id)
        self.personality = personality
        self.env = env
        self.player_analysis = (
            player_analysis if player_analysis is not None else {}
        )
        self.max_quantity = env.max_quantity

    def _make_fallback_bid(self, bot_dice: list) -> int:
        """Create minimal fallback bid."""
        face = bot_dice[0] if bot_dice[0] != 1 else 2
        return bid_to_action(1, face, self.max_quantity)

    def act(self, observation: Union[Dict, np.ndarray], deterministic: bool = False) -> int:
        """Choose action based on observation."""
        game_state = self.env.game_state

        if game_state.player_dice_count[self.agent_id] == 0:
            return 0

        bot_dice = game_state.get_player_dice(self.agent_id)
        current_bid = game_state.current_bid
        round_bid_history = game_state.bid_history.copy()

        try:
            if self.personality.name == BOT_PERSONALITIES["STANDARD_STAN"].name:
                decision, bid = get_standard_stan_decision(
                    game_state,
                    self.agent_id,
                    bot_dice,
                    self.player_analysis,
                    round_bid_history,
                )
            else:
                decision, bid = get_personality_decision(
                    game_state,
                    self.agent_id,
                    bot_dice,
                    self.personality,
                    self.player_analysis,
                    round_bid_history,
                )
        except (IndexError, ValueError):
            return self._make_fallback_bid(bot_dice) if current_bid is None else 0

        if decision == "DUDO":
            return 0
        elif decision == "CALZA":
            return 1
        elif decision == "BID" and bid is not None:
            quantity, face = bid
            return bid_to_action(quantity, face, self.max_quantity)
        else:
            return self._make_fallback_bid(bot_dice) if current_bid is None else 0

    def should_start_special_round(self, game_state: GameState) -> bool:
        """Determine if bot should start a special round."""
        bot_dice = game_state.get_player_dice(self.agent_id)
        total_dice_in_play = sum(game_state.player_dice_count)

        if self.personality.name == BOT_PERSONALITIES["STANDARD_STAN"].name:
            return should_stan_start_special_round(bot_dice, total_dice_in_play)
        else:
            return should_others_start_special_round(
                self.personality.name, total_dice_in_play
            )

    def learn(self, *args, **kwargs):
        """Train the agent (not applicable for rule-based bots)."""
        pass

    def reset(self):
        """Reset agent state."""
        self.player_analysis = {}
        for player_id in range(self.env.game_state.num_players):
            self.player_analysis[player_id] = create_initial_player_analysis(
                player_id
            )

