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
        self.special_round_active: bool = False  # Special round active when declared by player with 1 die
        self.special_round_declared_by: Optional[int] = None  # Player who declared special round
        self.special_round_used: List[bool] = [False] * num_players  # Track if player has used special round
        self.believe_called: bool = False  # Whether believe was called

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
        self.special_round_active = False
        self.special_round_declared_by = None
        self.special_round_used = [False] * self.num_players
        self.believe_called = False
        self.game_over = False
        self.winner = None

        # Roll dice for all players
        self.roll_dice()

    def roll_dice(self) -> None:
        """Roll dice for all players."""
        self.player_dice = []
        # Reset palifico status at the start of each round
        self.palifico_active = [count == 1 for count in self.player_dice_count]
        # Special round is NOT automatically activated - it must be declared
        # Reset special round at start of each round (unless continuing declared round)
        # (Special round persists until round ends)
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
        Get all dice from all players (for checking after believe).

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
        Check if bid (q1, v1) is higher than (q2, v2) according to new rules.

        New rules:
        1. Quantity can only increase (exception: can reduce by half when calling "ones")
        2. If quantity same, value must increase
        3. If quantity increases, any value can be called
        4. If previous bet was "ones": can increase quantity of ones OR use different value with quantity = 2 * prev_ones + 1

        Args:
            q1, v1: New bid
            q2, v2: Previous bid

        Returns:
            True if new bid is higher
        """
        # Special case: if previous bet was "ones" (v2 == 1)
        if v2 == 1:
            # New bid must be either:
            # - Increase quantity of ones (v1 == 1 and q1 > q2), OR
            # - Different value with quantity >= 2 * q2 + 1
            if v1 == 1:
                return q1 > q2  # Must increase quantity of ones
            else:
                # Different value, must have quantity >= 2 * q2 + 1
                required_quantity = 2 * q2 + 1
                return q1 >= required_quantity
        
        # Previous bet was not "ones"
        # If new bid is "ones", can reduce quantity by half (rounding up)
        if v1 == 1:
            # Can reduce by half (rounding up for odd numbers)
            # So minimum quantity = ceil(q2 / 2) = (q2 + 1) // 2
            required_quantity = (q2 + 1) // 2
            return q1 >= required_quantity
        
        # Both bids are not "ones"
        if q1 > q2:
            return True  # Quantity increased, any value allowed
        elif q1 == q2:
            return v1 > v2  # Same quantity, value must increase
        else:
            return False  # Quantity decreased (not allowed)

    def _count_dice_for_value(self, value: int) -> int:
        """
        Count dice with specified value, respecting special round rules.

        In special round: ones are NOT jokers (only count exact matches)
        In normal round: ones are jokers (count as any value except when bidding on 1s)

        Args:
            value: Dice value to count

        Returns:
            Total count of dice matching the value
        """
        total_count = 0
        for player_dice in self.player_dice:
            for die in player_dice:
                if self.special_round_active:
                    # In special round: ones are NOT jokers
                    if die == value:
                        total_count += 1
                else:
                    # In normal round: ones are jokers (except when bidding on 1s)
                    if die == value or (die == 1 and value != 1):
                        total_count += 1
        return total_count

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

        if not self.bid_history:
            return False, 0, 0

        # The player who made the bid is the last one in the history
        previous_player = self.bid_history[-1][0]

        # The challenger cannot be the one who made the bid
        if previous_player == challenger_id:
            # This case should not happen in a well-formed game
            return False, 0, 0

        quantity, value = self.current_bid

        # Count all dice with specified value (respecting special round rules)
        total_count = self._count_dice_for_value(value)

        # Challenge succeeds if actual count is less than bid
        challenge_success = total_count < quantity

        return challenge_success, total_count, quantity

    def call_believe(self, caller_id: int) -> Tuple[bool, int]:
        """
        Call believe - all players show their dice.

        Args:
            caller_id: ID of player calling believe

        Returns:
            Tuple (whether dice count exactly equals bid, actual dice count)
        """
        if self.current_bid is None:
            return False, 0

        quantity, value = self.current_bid

        # Count all dice (respecting special round rules)
        total_count = self._count_dice_for_value(value)

        # Believe succeeds if actual count exactly equals bid
        believe_success = total_count == quantity

        self.believe_called = True
        return believe_success, total_count

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

            # Check game over
            if self.player_dice_count[player_id] == 0:
                self._check_game_over()

    def gain_dice(self, player_id: int, count: int = 1) -> None:
        """
        Player gains dice (max 5).

        Args:
            player_id: Player ID
            count: Number of dice to gain
        """
        if 0 <= player_id < self.num_players:
            self.player_dice_count[player_id] = min(
                self.dice_per_player, self.player_dice_count[player_id] + count
            )

    def declare_special_round(self, player_id: int) -> bool:
        """
        Declare special round (can only be done once by player with 1 die, must have > 2 active players).

        Args:
            player_id: Player ID declaring special round

        Returns:
            True if special round was successfully declared
        """
        # Check conditions
        if self.player_dice_count[player_id] != 1:
            return False  # Must have exactly 1 die
        
        if self.special_round_used[player_id]:
            return False  # Can only be declared once
        
        # Count active players (players with dice)
        active_players = sum(1 for count in self.player_dice_count if count > 0)
        if active_players <= 2:
            return False  # Must have more than 2 active players
        
        # Declare special round
        self.special_round_active = True
        self.special_round_declared_by = player_id
        self.special_round_used[player_id] = True
        return True

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
        
        # Check if game should be over after skipping players
        # This ensures game_over is set correctly even if all but one player
        # were eliminated by skipping rather than by losing dice
        if not self.game_over:
            self._check_game_over()

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
            "special_round_active": self.special_round_active,
            "believe_called": self.believe_called,
            "game_over": self.game_over,
            "winner": self.winner,
        }
