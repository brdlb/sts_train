"""
Rule-based agent that uses bot logic from TypeScript implementation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from .base_agent import BaseAgent
from .bot_types import (
    BotPersonality,
    BotDecision,
    Bid,
    GameStage,
    PlayerAnalysis,
)
from .bot_genesis import (
    should_stan_start_special_round,
    get_standard_stan_decision,
)
from .bot_logic import (
    should_others_start_special_round,
    get_personality_decision,
)
from .bot_personalities import BOT_PERSONALITIES
from ..utils.helpers import bid_to_action, decode_bid


class RuleBasedAgent(BaseAgent):
    """
    Rule-based agent that uses deterministic bot logic.
    
    This agent converts observations to game state format expected by bot logic,
    makes decisions using the bot personality, and converts decisions back to actions.
    """

    def __init__(
        self,
        agent_id: int,
        personality: BotPersonality,
        max_quantity: int = 30,
        max_players: int = 8,
        max_history_length: int = 20,
    ):
        """
        Initialize rule-based agent.
        
        Args:
            agent_id: Unique agent ID
            personality: Bot personality configuration
            max_quantity: Maximum dice quantity in bids
            max_players: Maximum number of players
            max_history_length: Maximum bid history length
        """
        super().__init__(agent_id)
        self.personality = personality
        self.max_quantity = max_quantity
        self.max_players = max_players
        self.max_history_length = max_history_length
        
        # Track player analysis for all players
        # Format: {player_id: PlayerAnalysis}
        self.player_analysis: Dict[str, PlayerAnalysis] = {}
        
        # Track round bid history for current round
        # Format: List[{"bid": Bid, "bidder_id": str}]
        self.round_bid_history: List[Dict] = []
        
        # Track if we're in a new round (to reset round_bid_history)
        self.last_round_number: Optional[int] = None
        
        # Track previous observation for round result analysis
        self.previous_observation: Optional[Dict[str, np.ndarray]] = None

    def get_action(self, observation: Any, action_mask: Optional[np.ndarray] = None) -> int:
        """
        Get action from observation (alias for act, used by VecEnv).
        
        Args:
            observation: Observation from environment
            action_mask: Optional action mask (ignored for rule-based)
        
        Returns:
            Action from action_space
        """
        return self.act(observation, deterministic=True)
    
    def act(self, observation: Any, deterministic: bool = False) -> int:
        """
        Choose action based on observation.
        
        Args:
            observation: Observation from environment (Dict with 'bid_history', 'static_info', 'action_mask')
            deterministic: Whether to use deterministic policy (ignored for rule-based)
        
        Returns:
            Action from action_space
        """
        # Handle both Dict and array observations
        if not isinstance(observation, dict):
            # If array, convert to dict format (shouldn't happen with current env, but handle it)
            raise ValueError("RuleBasedAgent expects Dict observation, not array")
        
        # Convert observation to game state format
        game_state_data = self._convert_observation_to_game_state(observation)
        
        # Get bot decision
        if self.personality.name == BOT_PERSONALITIES["STANDARD_STAN"].name:
            decision = get_standard_stan_decision(
                current_bid=game_state_data["current_bid"],
                bot_dice=game_state_data["bot_dice"],
                total_dice_in_play=game_state_data["total_dice_in_play"],
                active_player_count=game_state_data["active_player_count"],
                is_special_round=game_state_data["is_special_round"],
                game_stage=game_state_data["game_stage"],
                round_bid_history=self.round_bid_history,
                player_analysis=self.player_analysis,
                last_bidder_id=game_state_data["last_bidder_id"],
                last_bidder_dice=game_state_data["last_bidder_dice"],
            )
        else:
            decision = get_personality_decision(
                current_bid=game_state_data["current_bid"],
                bot_dice=game_state_data["bot_dice"],
                bot_dice_count=game_state_data["bot_dice_count"],
                personality=self.personality,
                total_dice_in_play=game_state_data["total_dice_in_play"],
                active_player_count=game_state_data["active_player_count"],
                is_special_round=game_state_data["is_special_round"],
                game_stage=game_state_data["game_stage"],
                round_bid_history=self.round_bid_history,
                player_analysis=self.player_analysis,
                last_bidder_id=game_state_data["last_bidder_id"],
                last_bidder_dice=game_state_data["last_bidder_dice"],
                all_players_dice=game_state_data["all_players_dice"],
                all_player_ids=game_state_data["all_player_ids"],
            )
        
        # Convert decision to action
        action = self._convert_decision_to_action(decision, observation)
        return action

    def _convert_observation_to_game_state(
        self, observation: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Convert observation to game state format expected by bot logic.
        
        Args:
            observation: Observation dictionary from environment
        
        Returns:
            Dictionary with game state data
        """
        static_info = observation["static_info"]
        bid_history = observation["bid_history"]
        
        # Parse static_info
        # Format: [agent_id one-hot (max_players), current_bid (2), dice_count (max_players),
        #          current_player (1), palifico (max_players), believe (1), 
        #          special_round_active (1), round_number (1), player_dice (5)]
        offset = 0
        
        # Agent ID one-hot (max_players values)
        agent_id_onehot = static_info[offset : offset + self.max_players]
        agent_id = int(np.argmax(agent_id_onehot))
        offset += self.max_players
        
        # Current bid (2 values)
        current_bid_quantity = int(static_info[offset])
        current_bid_value = int(static_info[offset + 1])
        current_bid = (
            Bid(quantity=current_bid_quantity, face=current_bid_value)
            if current_bid_quantity > 0
            else None
        )
        offset += 2
        
        # Dice count for each player (max_players values)
        player_dice_count = [
            int(count) for count in static_info[offset : offset + self.max_players]
        ]
        offset += self.max_players
        
        # Current player (1 value)
        current_player = int(static_info[offset])
        offset += 1
        
        # Palifico flags (max_players values)
        palifico_active = [
            bool(flag) for flag in static_info[offset : offset + self.max_players]
        ]
        offset += self.max_players
        
        # Believe flag (1 value)
        believe_called = bool(static_info[offset])
        offset += 1
        
        # Special round active flag (1 value)
        is_special_round = bool(static_info[offset])
        offset += 1
        
        # Round number (1 value)
        round_number = int(static_info[offset])
        offset += 1
        
        # Current player's dice (5 values)
        bot_dice = [int(d) for d in static_info[offset : offset + 5] if d > 0]
        bot_dice_count = len(bot_dice)
        
        # Calculate total dice in play
        total_dice_in_play = sum(player_dice_count)
        
        # Count active players
        active_player_count = sum(1 for count in player_dice_count if count > 0)
        
        # Determine game stage
        from .bot_utils import get_game_stage
        
        game_stage = get_game_stage(total_dice_in_play, active_player_count)
        
        # Check if we're in a new round (to reset round_bid_history and analyze previous round)
        if self.last_round_number is not None and round_number != self.last_round_number:
            # New round started, analyze previous round result
            if self.previous_observation is not None:
                self._analyze_round_result(self.previous_observation, observation)
            # Reset round bid history
            self.round_bid_history = []
        self.last_round_number = round_number
        self.previous_observation = observation.copy() if isinstance(observation, dict) else observation
        
        # Parse bid history
        # Format: (max_history_length, 2) - (action_type, encoded_bid)
        # action_type: 0=bid, 1=challenge, 2=believe
        # encoded_bid: encoded quantity and value via encode_bid(quantity, value)
        # History is stored with most recent first
        # Only add bids from current round to round_bid_history
        round_bid_history_list = []
        last_bidder_index = None
        
        # Track player order for approximate player_id assignment
        # We'll use a simple approach: assume bids are made in player order
        # This is approximate but necessary since player_id is not in observation
        player_order_counter = 0
        
        for i in range(self.max_history_length):
            action_type = int(bid_history[i][0])
            encoded_bid = int(bid_history[i][1])
            
            # Only process bids (action_type == 0)
            if action_type == 0 and encoded_bid >= 0:
                # Decode bid
                quantity, value = decode_bid(encoded_bid, self.max_quantity)
                
                # Use approximate player_id based on order in history
                # This is not perfect but necessary since player_id is not available
                # We'll use a simple counter that cycles through players
                approximate_player_id = player_order_counter % self.max_players
                player_order_counter += 1
                
                # Add to round bid history (only if it's a new bid not already in history)
                bid_entry = {
                    "bid": Bid(quantity=quantity, face=value),
                    "bidder_id": f"player_{approximate_player_id}",
                }
                # Check if this bid is already in round_bid_history
                if not any(
                    entry["bid"].quantity == quantity and entry["bid"].face == value and entry["bidder_id"] == bid_entry["bidder_id"]
                    for entry in self.round_bid_history
                ):
                    round_bid_history_list.append(bid_entry)
                
                # Track last bidder (first valid entry is most recent)
                if last_bidder_index is None:
                    last_bidder_index = approximate_player_id
        
        # Update round bid history (append new bids, keep chronological order)
        # Reverse to get chronological order (oldest first)
        new_bids = list(reversed(round_bid_history_list))
        # Merge with existing round_bid_history, avoiding duplicates
        for bid_entry in new_bids:
            if bid_entry not in self.round_bid_history:
                self.round_bid_history.append(bid_entry)
        
        # Find last bidder
        last_bidder_id = None
        last_bidder_dice = []
        if last_bidder_index is not None:
            last_bidder_id = f"player_{last_bidder_index}"
            # We don't know other players' dice, so use empty list
            # Bot logic will use probabilities instead
            last_bidder_dice = []
        elif self.round_bid_history:
            # Fallback: use last entry in round_bid_history
            last_bid_data = self.round_bid_history[-1]
            last_bidder_id = last_bid_data["bidder_id"]
            last_bidder_dice = []
        
        # Create player IDs list
        all_player_ids = [f"player_{i}" for i in range(self.max_players)]
        
        # Create all players dice (we only know counts, not actual dice)
        all_players_dice = [
            [0] * count for count in player_dice_count
        ]  # Placeholder - actual dice unknown
        
        # Initialize player analysis if needed
        for player_id in all_player_ids:
            if player_id not in self.player_analysis:
                self.player_analysis[player_id] = PlayerAnalysis(player_id=player_id)
        
        return {
            "current_bid": current_bid,
            "bot_dice": bot_dice,
            "bot_dice_count": bot_dice_count,
            "total_dice_in_play": total_dice_in_play,
            "active_player_count": active_player_count,
            "is_special_round": is_special_round,
            "game_stage": game_stage,
            "last_bidder_id": last_bidder_id,
            "last_bidder_dice": last_bidder_dice,
            "all_players_dice": all_players_dice,
            "all_player_ids": all_player_ids,
        }

    def _convert_decision_to_action(
        self, decision: BotDecision, observation: Dict[str, np.ndarray]
    ) -> int:
        """
        Convert bot decision to action ID.
        
        Args:
            decision: Bot decision
            observation: Observation (for action mask)
        
        Returns:
            Action ID
        """
        if decision.decision == "DUDO":
            return 0  # Challenge action
        elif decision.decision == "CALZA":
            return 1  # Believe action
        elif decision.decision == "BID" and decision.bid:
            # Convert bid to action
            return bid_to_action(
                decision.bid.quantity, decision.bid.face, self.max_quantity
            )
        else:
            # Fallback: challenge if no valid decision
            return 0

    def _analyze_round_result(
        self,
        previous_obs: Dict[str, np.ndarray],
        current_obs: Dict[str, np.ndarray],
    ) -> None:
        """
        Analyze round result when a new round starts.
        
        This method is called automatically when round_number changes.
        It analyzes the previous round to update player analysis statistics.
        
        Args:
            previous_obs: Observation from previous step (end of previous round)
            current_obs: Observation from current step (start of new round)
        """
        # Extract information from previous observation
        prev_static = previous_obs["static_info"]
        prev_bid_history = previous_obs["bid_history"]
        
        # Find the last bid from previous round
        # Format: (action_type, encoded_bid)
        last_bid = None
        last_bidder_id = None
        for i in range(self.max_history_length):
            action_type = int(prev_bid_history[i][0])
            encoded_bid = int(prev_bid_history[i][1])
            
            # Only process bids (action_type == 0)
            if action_type == 0 and encoded_bid >= 0:
                # Decode bid
                quantity, value = decode_bid(encoded_bid, self.max_quantity)
                # Use approximate player_id (we don't have exact player_id in observation)
                approximate_player_id = 0  # Default to 0, actual player_id not available
                last_bid = {"quantity": quantity, "value": value, "player_id": approximate_player_id}
                last_bidder_id = f"player_{approximate_player_id}"
                break
        
        if last_bid is None:
            return  # No bid to analyze
        
        # Check if this was the first bid of the round
        # Count how many bids were in the round
        round_bid_count = len(self.round_bid_history)
        was_first_bid = round_bid_count == 1
        
        # Determine if it was a pre-reveal bid (one of the last few bids before challenge/believe)
        # This is a heuristic: if there were few bids, it's likely pre-reveal
        was_pre_reveal = round_bid_count <= 3
        
        # We can't determine if it was a bluff without knowing actual dice counts
        # But we can track that a bid was made, and update statistics when we have more info
        # For now, we'll just track that bids were made
        
        # Update analysis for the bidder
        if last_bidder_id:
            # We don't know if it was a bluff, but we can track the bid
            # The actual bluff detection would require knowing the actual dice count
            # which is not available in observation
            pass  # Placeholder - actual analysis requires dice information
    
    def _update_player_analysis(
        self,
        player_id: str,
        analysis_type: str,
        data: Dict[str, Any],
    ) -> None:
        """
        Update player analysis statistics.
        
        Args:
            player_id: Player ID
            analysis_type: Type of analysis ('first_bid', 'pre_reveal', 'face_pattern')
            data: Analysis data dictionary
        """
        if player_id not in self.player_analysis:
            self.player_analysis[player_id] = PlayerAnalysis(player_id=player_id)
        
        analysis = self.player_analysis[player_id]
        
        if analysis_type == 'first_bid':
            was_bluff = data.get('was_bluff', False)
            analysis.first_bid_bluffs.total += 1
            if was_bluff:
                analysis.first_bid_bluffs.count += 1
        elif analysis_type == 'pre_reveal':
            was_bluff = data.get('was_bluff', False)
            was_strong_hand = data.get('was_strong_hand', False)
            analysis.pre_reveal_tendency.total += 1
            if was_bluff:
                analysis.pre_reveal_tendency.bluff_count += 1
            if was_strong_hand:
                analysis.pre_reveal_tendency.strong_hand_count += 1
        elif analysis_type == 'face_pattern':
            bid_face = data.get('bid_face')
            was_bluff = data.get('was_bluff', False)
            if bid_face is not None:
                face_key = str(bid_face)
                if face_key not in analysis.face_bluff_patterns:
                    from .bot_types import FaceBluffPattern
                    analysis.face_bluff_patterns[face_key] = FaceBluffPattern()
                
                pattern = analysis.face_bluff_patterns[face_key]
                pattern.total_bids += 1
                if was_bluff:
                    pattern.bluff_count += 1
    
    def update_player_analysis(
        self,
        player_id: str,
        was_first_bid_bluff: Optional[bool] = None,
        was_pre_reveal_bluff: Optional[bool] = None,
        was_pre_reveal_strong_hand: Optional[bool] = None,
        bid_face: Optional[int] = None,
    ):
        """
        Update player analysis statistics.
        
        This should be called after a round ends to update analysis data.
        
        Args:
            player_id: Player ID
            was_first_bid_bluff: Whether first bid was a bluff
            was_pre_reveal_bluff: Whether pre-reveal bid was a bluff
            was_pre_reveal_strong_hand: Whether pre-reveal bid was strong hand
            bid_face: Face value of the bid (for face pattern analysis)
        """
        if player_id not in self.player_analysis:
            self.player_analysis[player_id] = PlayerAnalysis(player_id=player_id)
        
        analysis = self.player_analysis[player_id]
        
        # Update first bid bluff stats
        if was_first_bid_bluff is not None:
            analysis.first_bid_bluffs.total += 1
            if was_first_bid_bluff:
                analysis.first_bid_bluffs.count += 1
        
        # Update pre-reveal tendency stats
        if was_pre_reveal_bluff is not None or was_pre_reveal_strong_hand is not None:
            analysis.pre_reveal_tendency.total += 1
            if was_pre_reveal_bluff:
                analysis.pre_reveal_tendency.bluff_count += 1
            if was_pre_reveal_strong_hand:
                analysis.pre_reveal_tendency.strong_hand_count += 1
        
        # Update face bluff pattern
        if bid_face is not None:
            face_key = str(bid_face)
            if face_key not in analysis.face_bluff_patterns:
                from .bot_types import FaceBluffPattern
                analysis.face_bluff_patterns[face_key] = FaceBluffPattern()
            
            pattern = analysis.face_bluff_patterns[face_key]
            pattern.total_bids += 1
            if was_pre_reveal_bluff:
                pattern.bluff_count += 1

    def reset(self):
        """Reset agent state."""
        self.player_analysis = {}
        self.round_bid_history = []
        self.last_round_number = None
        self.previous_observation = None

    def learn(self, *args, **kwargs):
        """Rule-based agents don't learn."""
        pass

