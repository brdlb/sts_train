"""
Implementation of Perudo game rules.
"""

from typing import List, Tuple, Optional
from .game_state import GameState


class PerudoRules:
    """Class for working with Perudo rules."""

    @staticmethod
    def is_valid_bid(
        game_state: GameState,
        player_id: int,
        quantity: int,
        value: int,
    ) -> Tuple[bool, str]:
        """
        Check bid validity.

        Args:
            game_state: Current game state
            player_id: Player ID
            quantity: Number of dice
            value: Dice value

        Returns:
            Tuple (whether bid is valid, error message)
        """
        # Check that it's player's turn
        if game_state.current_player != player_id:
            return False, "Not your turn"

        # Check ranges
        if quantity <= 0:
            return False, "Quantity must be positive"
        if value < 1 or value > game_state.total_dice_values:
            return False, f"Value must be between 1 and {game_state.total_dice_values}"

        # Check that player is still in game
        if game_state.player_dice_count[player_id] == 0:
            return False, "Player already out of game"

        # Check that bid is higher than previous
        if game_state.current_bid is not None:
            prev_quantity, prev_value = game_state.current_bid
            if not game_state._is_bid_higher(quantity, value, prev_quantity, prev_value):
                return (
                    False,
                    f"Bid must be higher than previous ({prev_quantity}x{prev_value})",
                )
            # Palifico rule
            if game_state.palifico_active[player_id]:
                if value != prev_value:
                    return False, "Palifico player cannot change value"

        # Check maximum quantity (cannot exceed total dice count)
        total_dice = sum(game_state.player_dice_count)
        max_possible = total_dice * 2  # Considering pasari
        if quantity > max_possible:
            return False, f"Quantity cannot exceed {max_possible}"

        return True, ""

    @staticmethod
    def can_challenge(game_state: GameState, player_id: int) -> Tuple[bool, str]:
        """
        Check if player can challenge previous player.

        Args:
            game_state: Current game state
            player_id: Player ID

        Returns:
            Tuple (whether can challenge, error message)
        """
        if game_state.current_player != player_id:
            return False, "Not your turn"

        if game_state.current_bid is None:
            return False, "No bid to challenge"

        if game_state.player_dice_count[player_id] == 0:
            return False, "Player already out of game"

        # Check that there is a previous bid
        if len(game_state.bid_history) == 0:
            return False, "No previous bid"

        return True, ""

    @staticmethod
    def can_call_pacao(game_state: GameState, player_id: int) -> Tuple[bool, str]:
        """
        Check if player can call pacao.

        Args:
            game_state: Current game state
            player_id: Player ID

        Returns:
            Tuple (whether can call pacao, error message)
        """
        if game_state.current_player != player_id:
            return False, "Not your turn"

        if game_state.current_bid is None:
            return False, "No bid for pacao"

        if game_state.pacao_called:
            return False, "Pacao already called"

        if game_state.player_dice_count[player_id] != 1:
            return False, "Only players with one die can call pacao"

        return True, ""

    @staticmethod
    def process_challenge_result(
        game_state: GameState,
        challenger_id: int,
        challenge_success: bool,
        actual_count: int,
        bid_quantity: int,
    ) -> Tuple[int, int]:
        """
        Process challenge result.

        Args:
            game_state: Current game state
            challenger_id: ID of player who challenged
            challenge_success: Whether challenge succeeded (True if bid was wrong)
            actual_count: Actual dice count
            bid_quantity: Quantity in bid

        Returns:
            Tuple (ID of player who loses die, number of dice lost)
        """
        # The player who made the bid is the last one in the history
        if not game_state.bid_history:
            return challenger_id, 1

        previous_player = game_state.bid_history[-1][0]

        # If challenge succeeded (bid was wrong), player who made bid loses die
        # If challenge failed (bid was correct), challenger loses die
        if challenge_success:
            loser = previous_player
        else:
            loser = challenger_id

        return loser, 1

    @staticmethod
    def process_pacao_result(
        game_state: GameState,
        caller_id: int,
        pacao_success: bool,
        actual_count: int,
        bid_quantity: int,
    ) -> Tuple[int, int]:
        """
        Process pacao result.

        Args:
            game_state: Current game state
            caller_id: ID of player who called pacao
            pacao_success: Whether pacao succeeded (True if actual >= bid)
            actual_count: Actual dice count
            bid_quantity: Quantity in bid

        Returns:
            Tuple (ID of player who loses die, number of dice lost)
        """
        if not game_state.bid_history:
            return caller_id, 1

        previous_player = game_state.bid_history[-1][0]

        # If pacao succeeded (actual >= bid), player who made bid loses die
        # If pacao failed (actual < bid), caller loses die
        if pacao_success:
            loser = previous_player
        else:
            loser = caller_id

        return loser, 1

    @staticmethod
    def get_available_actions(
        game_state: GameState, player_id: int
    ) -> List[Tuple[str, Optional[int], Optional[int]]]:
        """
        Get list of available actions for player.

        Args:
            game_state: Current game state
            player_id: Player ID

        Returns:
            List of available actions in format (action_type, param1, param2)
            Types: 'bid', 'challenge', 'pacao'
        """
        actions = []

        # Check that it's player's turn
        if game_state.current_player != player_id:
            return actions

        if game_state.player_dice_count[player_id] == 0:
            return actions

        # Action: challenge previous player
        can_challenge, _ = PerudoRules.can_challenge(game_state, player_id)
        if can_challenge:
            actions.append(("challenge", None, None))

        # Action: call pacao
        can_pacao, _ = PerudoRules.can_call_pacao(game_state, player_id)
        if can_pacao:
            actions.append(("pacao", None, None))

        # Actions: bids
        # Generate possible bids
        if game_state.current_bid is None:
            # First bid - can be any
            min_quantity = 1
            max_quantity = sum(game_state.player_dice_count) * 2  # Considering pasari
            for q in range(min_quantity, min(max_quantity + 1, 30)):  # Limit to reasonable maximum
                for v in range(1, game_state.total_dice_values + 1):
                    actions.append(("bid", q, v))
        else:
            # Subsequent bids must be higher
            prev_quantity, prev_value = game_state.current_bid
            max_quantity = sum(game_state.player_dice_count) * 2
            for q in range(prev_quantity, min(max_quantity + 1, 30)):
                for v in range(1, game_state.total_dice_values + 1):
                    if game_state._is_bid_higher(q, v, prev_quantity, prev_value):
                        actions.append(("bid", q, v))

        return actions
