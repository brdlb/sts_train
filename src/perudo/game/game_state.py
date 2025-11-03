"""
Game state class for Perudo game.
"""

from typing import List, Optional, Tuple
import numpy as np


class GameState:
    """Class for managing Perudo game state."""

    def __init__(
        self,
        num_players: int,
        dice_per_player: int = 5,
        total_dice_values: int = 6,
    ):
        """
        Initialize game state.

        Args:
            num_players: Number of players
            dice_per_player: Number of dice per player
            total_dice_values: Total possible dice values (usually 6)
        """
        self.num_players = num_players
        self.dice_per_player = dice_per_player
        self.total_dice_values = total_dice_values

        # Hidden dice of all players (not visible to others)
        # Format: list of lists, where each inner list is a player's dice
        self.player_dice: List[List[int]] = []

        # Public information
        self.current_bid: Optional[Tuple[int, int]] = None  # (quantity, value)
        self.bid_history: List[Tuple[int, int, int]] = []  # (player, quantity, value)
        self.current_player: int = 0

        # Number of dice for each player (public information)
        self.player_dice_count: List[int] = [dice_per_player] * num_players

        # Game modes
        self.palifico_active: List[bool] = [False] * num_players  # Palifico active for player
        self.pacao_called: bool = False  # Whether pacao was called

        # Game status
        self.game_over: bool = False
        self.winner: Optional[int] = None

        # Initialize initial state
        self.reset()

    def reset(self) -> None:
        """Reset game to initial state."""
        self.player_dice = []
        self.current_bid = None
        self.bid_history = []
        self.current_player = 0
        self.player_dice_count = [self.dice_per_player] * self.num_players
        self.palifico_active = [False] * self.num_players
        self.pacao_called = False
        self.game_over = False
        self.winner = None

        # Roll dice for all players
        self.roll_dice()

    def roll_dice(self) -> None:
        """Roll dice for all players."""
        self.player_dice = []
        for player_id in range(self.num_players):
            dice = np.random.randint(1, self.total_dice_values + 1, size=self.player_dice_count[player_id]).tolist()
            self.player_dice.append(dice)

    def get_player_dice(self, player_id: int) -> List[int]:
        """
        Get dice for a specific player.

        Args:
            player_id: Player ID

        Returns:
            List of dice values for the player
        """
        if 0 <= player_id < self.num_players:
            return self.player_dice[player_id].copy()
        return []

    def get_all_dice(self) -> List[List[int]]:
        """
        Get all dice from all players (for checking after pacao).

        Returns:
            List of lists of dice from all players
        """
        return [dice.copy() for dice in self.player_dice]

    def set_bid(self, player_id: int, quantity: int, value: int) -> bool:
        """
        Set a bid.

        Args:
            player_id: ID of player making the bid
            quantity: Number of dice
            value: Dice value (1-6, where 1 is pasari)

        Returns:
            True if bid is valid and set
        """
        if self.current_player != player_id:
            return False

        if quantity <= 0 or value < 1 or value > self.total_dice_values:
            return False

        # Check that bid is higher than previous
        if self.current_bid is not None:
            prev_quantity, prev_value = self.current_bid
            if not self._is_bid_higher(quantity, value, prev_quantity, prev_value):
                return False

        self.current_bid = (quantity, value)
        self.bid_history.append((player_id, quantity, value))
        return True

    def _is_bid_higher(self, q1: int, v1: int, q2: int, v2: int) -> bool:
        """
        Check if bid (q1, v1) is higher than (q2, v2).

        In Perudo: higher quantity OR same quantity but higher value.
        Special rule: 1 (pasari) counts as "double" of any other value.

        Args:
            q1, v1: New bid
            q2, v2: Previous bid

        Returns:
            True if new bid is higher
        """
        # Pasari (1) counts as double value
        # Double the quantity for pasari
        effective_q1 = q1 * 2 if v1 == 1 else q1
        effective_q2 = q2 * 2 if v2 == 1 else q2

        if effective_q1 > effective_q2:
            return True
        if effective_q1 == effective_q2 and v1 > v2:
            return True
        return False

    def challenge_bid(self, challenger_id: int) -> Tuple[bool, int, int]:
        """
        Challenge previous player (challenge bid).

        Args:
            challenger_id: ID of player challenging

        Returns:
            Tuple (whether challenge succeeded, actual dice count, bid quantity)
        """
        if self.current_bid is None:
            return False, 0, 0

        # Find previous player (who made the bid)
        previous_player = None
        for i in range(len(self.bid_history) - 1, -1, -1):
            player_id, _, _ = self.bid_history[i]
            if player_id != challenger_id:
                previous_player = player_id
                break

        if previous_player is None:
            return False, 0, 0

        quantity, value = self.current_bid

        # Count all dice with specified value (including pasari)
        total_count = 0
        for player_dice in self.player_dice:
            for die in player_dice:
                if die == value or die == 1:  # Pasari counts as any value
                    total_count += 1

        # Challenge succeeds if actual count is less than bid
        challenge_success = total_count < quantity

        return challenge_success, total_count, quantity

    def call_pacao(self, caller_id: int) -> Tuple[bool, int]:
        """
        Call pacao - all players show their dice.

        Args:
            caller_id: ID of player calling pacao

        Returns:
            Tuple (whether pacao succeeded, actual dice count)
        """
        if self.current_bid is None:
            return False, 0

        quantity, value = self.current_bid

        # Count all dice
        total_count = 0
        for player_dice in self.player_dice:
            for die in player_dice:
                if die == value or die == 1:  # Pasari counts as any value
                    total_count += 1

        # Pacao succeeds if actual count is greater than or equal to bid
        pacao_success = total_count >= quantity

        self.pacao_called = True
        return pacao_success, total_count

    def lose_dice(self, player_id: int, count: int = 1) -> None:
        """
        Player loses dice.

        Args:
            player_id: Player ID
            count: Number of dice to lose
        """
        if 0 <= player_id < self.num_players:
            self.player_dice_count[player_id] = max(
                0, self.player_dice_count[player_id] - count
            )

            # If player has 1 die left, activate palifico
            if self.player_dice_count[player_id] == 1:
                self.palifico_active[player_id] = True

            # Check game over
            if self.player_dice_count[player_id] == 0:
                self._check_game_over()

    def next_player(self) -> None:
        """Move to next player."""
        self.current_player = (self.current_player + 1) % self.num_players

        # Skip players with no dice
        max_attempts = self.num_players
        attempts = 0
        while (
            self.player_dice_count[self.current_player] == 0
            and attempts < max_attempts
        ):
            self.current_player = (self.current_player + 1) % self.num_players
            attempts += 1

    def _check_game_over(self) -> None:
        """Check if game is over."""
        players_with_dice = sum(1 for count in self.player_dice_count if count > 0)
        if players_with_dice == 1:
            self.game_over = True
            for i, count in enumerate(self.player_dice_count):
                if count > 0:
                    self.winner = i
                    break

    def get_public_info(self) -> dict:
        """
        Get public information about game state.

        Returns:
            Dictionary with public information
        """
        return {
            "current_bid": self.current_bid,
            "bid_history": self.bid_history.copy(),
            "current_player": self.current_player,
            "player_dice_count": self.player_dice_count.copy(),
            "palifico_active": self.palifico_active.copy(),
            "pacao_called": self.pacao_called,
            "game_over": self.game_over,
            "winner": self.winner,
        }
