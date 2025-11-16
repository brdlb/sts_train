"""
Utility functions for bot logic.
"""

from typing import List, Tuple, Dict, Literal, Optional
import numpy as np
from ...game.game_state import GameState
from .player_analysis import PlayerAnalysis

GameStage = Literal["CHAOS", "POSITIVE", "TENSE", "KNIFE_FIGHT", "DUEL"]


def format_bid_face(face: int) -> str:
    """
    Format dice face for display.

    Args:
        face: Dice face value (1-6)

    Returns:
        Formatted string (★ for 1, number otherwise)
    """
    return "★" if face == 1 else str(face)


def format_bid(quantity: int, face: int) -> str:
    """
    Format bid for display.

    Args:
        quantity: Number of dice
        face: Dice face value

    Returns:
        Formatted bid string
    """
    return f"{quantity} x {format_bid_face(face)}"


def get_game_stage(total_dice_in_play: int, active_player_count: int) -> GameStage:
    """
    Determine the current stage of the game based on total dice and active players.

    Args:
        total_dice_in_play: Total number of dice in play
        active_player_count: Number of active players

    Returns:
        Game stage
    """
    if active_player_count == 2:
        return "DUEL"
    if total_dice_in_play >= 26:
        return "CHAOS"
    if total_dice_in_play >= 13:
        return "POSITIVE"
    if total_dice_in_play >= 6:
        return "TENSE"
    return "KNIFE_FIGHT"  # 5 or fewer dice


def calculate_expected_count(
    face: int,
    bot_dice: List[int],
    total_dice_in_play: int,
    is_special_round: bool,
) -> float:
    """
    Calculate the statistically expected number of dice for a given face.

    Args:
        face: Dice face value (1-6)
        bot_dice: Bot's dice
        total_dice_in_play: Total number of dice in play
        is_special_round: Whether special round is active

    Returns:
        Expected count of dice with given face
    """
    unknown_dice_count = total_dice_in_play - len(bot_dice)
    # In a special round, or when bidding on 1s, they are not wild.
    if is_special_round or face == 1:
        count_in_hand = bot_dice.count(face)
        expected_from_others = unknown_dice_count / 6
    else:
        # In normal round: ones are jokers (count as any value except when bidding on 1s)
        count_in_hand = bot_dice.count(face) + bot_dice.count(1)
        expected_from_others = unknown_dice_count / 3
    return count_in_hand + expected_from_others


def get_hand_strength(bot_dice: List[int], is_special_round: bool) -> Dict[int, int]:
    """
    Count the occurrences of each face in a bot's hand, including wilds.

    Args:
        bot_dice: Bot's dice
        is_special_round: Whether special round is active

    Returns:
        Dictionary mapping face (1-6) to count (including wilds)
    """
    counts: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    wilds = 0 if is_special_round else bot_dice.count(1)
    
    for die in bot_dice:
        counts[die] += 1
    
    if not is_special_round:
        # Add wilds to all faces except 1
        for face in range(2, 7):
            counts[face] += wilds
    
    return counts


def generate_possible_next_bids(
    current_bid: Tuple[int, int],
    total_dice_in_play: int,
    is_special_round: bool,
) -> List[Tuple[int, int]]:
    """
    Generate a list of all valid next bids.

    Args:
        current_bid: Current bid (quantity, face)
        total_dice_in_play: Total number of dice in play
        is_special_round: Whether special round is active

    Returns:
        List of possible next bids as (quantity, face) tuples
    """
    options: List[Tuple[int, int]] = []
    quantity, face = current_bid

    # In a special round, only quantity can be increased.
    if is_special_round:
        for q in range(quantity + 1, total_dice_in_play + 1):
            options.append((q, face))
        return options

    if face == 1:
        # RULE: Switching FROM 1s (Pacos)
        required_quantity = quantity * 2 + 1
        if required_quantity <= total_dice_in_play:
            for f in range(2, 7):
                options.append((required_quantity, f))
    else:
        # RULE: Standard bidding (current bid is not on 1s)
        
        # 1. Increase quantity on the same face
        for q in range(quantity + 1, total_dice_in_play + 1):
            options.append((q, face))
        
        # 2. Change to a higher face with the same quantity
        for f in range(face + 1, 7):
            options.append((quantity, f))
        
        # 3. Switch TO 1s (Pacos)
        required_ones_quantity = (quantity + 1) // 2  # ceil(quantity / 2)
        if required_ones_quantity > 0:
            # Can bid more than the minimum required
            for q in range(required_ones_quantity, total_dice_in_play + 1):
                options.append((q, 1))

    # Remove duplicates and filter invalid bids
    unique_options = list(set(options))
    return [
        (q, f)
        for q, f in unique_options
        if q > 0 and q <= total_dice_in_play and 1 <= f <= 6
    ]


def apply_pre_reveal_analysis(
    initial_expected_count: float,
    game_state: GameState,
    bot_id: int,
    bot_dice: List[int],
    last_bidder_id: int,
    last_bidder_dice: List[int],
    analysis: PlayerAnalysis,
    bot_affinity: float = 1.0,
) -> float:
    """
    Apply analysis of a player's pre-reveal tendencies to adjust the expected count of a bid.

    Args:
        initial_expected_count: Initial expected count
        game_state: Current game state
        bot_id: Bot's player ID
        bot_dice: Bot's dice
        last_bidder_id: Last bidder's player ID
        last_bidder_dice: Last bidder's dice
        analysis: Player analysis for last bidder
        bot_affinity: Bot's affinity for pre-reveal analysis

    Returns:
        Adjusted expected count
    """
    # We need a reasonable amount of data to make a judgment.
    if not analysis or analysis.pre_reveal_tendency.total < 3:
        return initial_expected_count

    pre_reveal = analysis.pre_reveal_tendency
    bluff_ratio = pre_reveal.bluff_count / pre_reveal.total if pre_reveal.total > 0 else 0.0
    strong_hand_ratio = pre_reveal.strong_hand_count / pre_reveal.total if pre_reveal.total > 0 else 0.0
    
    if game_state.current_bid is None:
        return initial_expected_count

    current_bid_quantity, current_bid_face = game_state.current_bid
    is_special_round = game_state.special_round_active
    
    total_dice_in_play = sum(game_state.player_dice_count)
    active_player_count = sum(1 for count in game_state.player_dice_count if count > 0)
    game_stage = get_game_stage(total_dice_in_play, active_player_count)

    # To avoid being predictable, the analysis is capped. The cap is lower in early game.
    max_effectiveness = 0.8  # Default cap for TENSE, KNIFE_FIGHT, DUEL
    if game_stage == "CHAOS":
        max_effectiveness = 0.6
    elif game_stage == "POSITIVE":
        max_effectiveness = 0.7

    # --- Scenario 1: Handle "Known Bluffers" ---
    bluff_threshold = 0.40
    high_bluff_threshold = 0.70

    if bluff_ratio > bluff_threshold:
        # Calculate the "pessimistic" scenario where the bidder has NO relevant dice.
        if is_special_round or current_bid_face == 1:
            bot_hand_count = bot_dice.count(current_bid_face)
            probability = 1.0 / 6
        else:
            bot_hand_count = bot_dice.count(current_bid_face) + bot_dice.count(1)
            probability = 1.0 / 3
        
        dice_excluding_bot_and_bidder = max(
            0, total_dice_in_play - len(bot_dice) - len(last_bidder_dice)
        )
        expected_from_others_excluding_bidder = dice_excluding_bot_and_bidder * probability
        pessimistic_expected_count = bot_hand_count + expected_from_others_excluding_bidder

        doubt_factor = 0.0
        if bluff_ratio >= high_bluff_threshold:
            # Player is a confirmed liar, we fully distrust them.
            doubt_factor = 1.0
        else:
            # Player is a suspected bluffer. We scale our doubt based on how far they are into the "unknown" zone.
            doubt_factor = (bluff_ratio - bluff_threshold) / (
                high_bluff_threshold - bluff_threshold
            )

        doubt_factor *= bot_affinity  # The bot's personality affects how much it trusts this analysis.
        doubt_factor = max(0.0, min(max_effectiveness, doubt_factor))  # Clamp with the dynamic cap.

        # The final expected count is a weighted average between the normal and pessimistic scenarios.
        return (initial_expected_count * (1 - doubt_factor)) + (
            pessimistic_expected_count * doubt_factor
        )

    # --- Scenario 2: Handle "Reliable Players" ---
    strong_hand_threshold = 0.40
    high_strong_hand_threshold = 0.70

    if strong_hand_ratio > strong_hand_threshold:
        trust_factor = 0.0
        if strong_hand_ratio >= high_strong_hand_threshold:
            trust_factor = 1.0
        else:
            trust_factor = (strong_hand_ratio - strong_hand_threshold) / (
                high_strong_hand_threshold - strong_hand_threshold
            )

        trust_factor *= bot_affinity
        trust_factor = max(0.0, min(max_effectiveness, trust_factor))  # Clamp with the dynamic cap.

        # Add a bonus to the expected count, making the bid seem more plausible.
        trust_bonus = trust_factor * 0.75  # Max bonus of +0.75 to expected count.
        return initial_expected_count + trust_bonus

    # If neither tendency is strong enough, return the original calculation.
    return initial_expected_count


def get_first_bid_adjustment(analysis: Optional[PlayerAnalysis]) -> float:
    """
    Get adjustment based on first bid bluff analysis.

    Args:
        analysis: Player analysis

    Returns:
        Adjustment value (0 if not enough data)
    """
    # Is the current bidder known for bluffing their opening bids?
    if not analysis or analysis.first_bid_bluffs.total < 2:
        return 0.0  # Not enough data
    
    ratio = (
        analysis.first_bid_bluffs.count / analysis.first_bid_bluffs.total
        if analysis.first_bid_bluffs.total > 0
        else 0.0
    )
    if ratio > 0.70:
        return 1.5  # High mistrust -> High adjustment
    if ratio > 0.45:
        return 0.7  # Mild mistrust -> Mild adjustment
    return 0.0


def get_face_pattern_adjustment(analysis: Optional[PlayerAnalysis], bid_face: int) -> float:
    """
    Get adjustment based on face pattern analysis.

    Args:
        analysis: Player analysis
        bid_face: Dice face being bid on

    Returns:
        Adjustment value
    """
    # Does the current bidder have a habit of bluffing on this specific face?
    if not analysis:
        return 0.0
    
    pattern = analysis.face_bluff_patterns.get(bid_face)
    if not pattern or pattern.total_bids < 1:
        return 0.0
    
    ratio = pattern.bluff_count / pattern.total_bids if pattern.total_bids > 0 else 0.0
    if ratio > 0.75:
        return 2.0  # Very strong signal
    if ratio > 0.50:
        return 1.0  # Strong signal
    return 0.0

