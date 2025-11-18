"""
Utility functions for rule-based bot logic.
"""

import random
from typing import List, Optional, Dict, Tuple
import numpy as np

from .bot_types import (
    GameStage,
    Bid,
    PlayerAnalysis,
    PreRevealTendency,
    FaceBluffPattern,
)


def format_bid_face(face: int) -> str:
    """
    Format dice face for display.
    
    Args:
        face: Dice value (1-6)
    
    Returns:
        Formatted string ("★" for 1, number for others)
    """
    return "★" if face == 1 else str(face)


def format_bid(bid: Bid) -> str:
    """
    Format bid for display.
    
    Args:
        bid: Bid to format
    
    Returns:
        Formatted string (e.g., "3 x ★" or "5 x 4")
    """
    return f"{bid.quantity} x {format_bid_face(bid.face)}"


def get_game_stage(total_dice_in_play: int, active_player_count: int) -> GameStage:
    """
    Determine the current stage of the game based on total dice and active players.
    
    Args:
        total_dice_in_play: Total number of dice across all players
        active_player_count: Number of players with dice remaining
    
    Returns:
        GameStage enum value
    """
    if active_player_count == 2:
        return GameStage.DUEL
    if total_dice_in_play >= 26:
        return GameStage.CHAOS
    if total_dice_in_play >= 13:
        return GameStage.POSITIVE
    if total_dice_in_play >= 6:
        return GameStage.TENSE
    return GameStage.KNIFE_FIGHT  # 5 or fewer dice


def calculate_expected_count(
    face: int,
    bot_dice: List[int],
    total_dice_in_play: int,
    is_special_round: bool,
) -> float:
    """
    Calculate the statistically expected number of dice for a given face.
    
    Now considers if it's a special round where 1s are not wild.
    
    Args:
        face: Dice value to count (1-6)
        bot_dice: Bot's dice
        total_dice_in_play: Total dice across all players
        is_special_round: Whether special round is active
    
    Returns:
        Expected count of dice matching the face
    """
    unknown_dice_count = total_dice_in_play - len(bot_dice)
    
    # In a special round, or when bidding on 1s, they are not wild.
    if is_special_round or face == 1:
        count_in_hand = sum(1 for d in bot_dice if d == face)
    else:
        # In normal round: count face OR 1 (joker)
        count_in_hand = sum(1 for d in bot_dice if d == face or d == 1)
    
    # Probability of getting the face from unknown dice
    if is_special_round or face == 1:
        probability = 1.0 / 6.0
    else:
        probability = 1.0 / 3.0  # Face or 1 (joker)
    
    expected_from_others = unknown_dice_count * probability
    return count_in_hand + expected_from_others


def get_hand_strength(bot_dice: List[int], is_special_round: bool) -> Dict[int, int]:
    """
    Count the occurrences of each face in a bot's hand, including wilds.
    
    Args:
        bot_dice: Bot's dice
        is_special_round: Whether special round is active
    
    Returns:
        Dictionary mapping dice value (1-6) to count (including wilds for non-1s)
    """
    counts: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    
    # Count natural occurrences
    for die in bot_dice:
        counts[die] += 1
    
    # In special round, 1s are not wild
    if not is_special_round:
        wilds = counts[1]
        # Add wilds to all non-1 faces
        for face in range(2, 7):
            counts[face] += wilds
    
    return counts


def generate_possible_next_bids(
    current_bid: Bid,
    total_dice_in_play: int,
    is_special_round: bool,
) -> List[Bid]:
    """
    Generate a list of all valid next bids.
    
    Now handles special round rules (no face change).
    
    Args:
        current_bid: Current bid
        total_dice_in_play: Total dice in game
        is_special_round: Whether special round is active
    
    Returns:
        List of valid next bids
    """
    options: List[Bid] = []
    quantity = current_bid.quantity
    face = current_bid.face
    
    # In a special round, only quantity can be increased.
    if is_special_round:
        for q in range(quantity + 1, total_dice_in_play + 1):
            options.append(Bid(quantity=q, face=face))
        return options
    
    if face == 1:
        # RULE: Switching FROM 1s (Pacos)
        required_quantity = quantity * 2 + 1
        if required_quantity <= total_dice_in_play:
            for f in range(2, 7):
                options.append(Bid(quantity=required_quantity, face=f))
    else:
        # RULE: Standard bidding (current bid is not on 1s)
        
        # 1. Increase quantity on the same face
        for q in range(quantity + 1, total_dice_in_play + 1):
            options.append(Bid(quantity=q, face=face))
        
        # 2. Change to a higher face with the same quantity
        for f in range(face + 1, 7):
            options.append(Bid(quantity=quantity, face=f))
        
        # 3. Switch TO 1s (Pacos)
        required_ones_quantity = (quantity + 1) // 2  # Ceil division
        if required_ones_quantity > 0:
            # Can bid more than the minimum required
            for q in range(required_ones_quantity, total_dice_in_play + 1):
                options.append(Bid(quantity=q, face=1))
    
    # Remove duplicates and filter valid bids
    seen = set()
    valid_bids = []
    for bid in options:
        if bid.quantity > 0 and bid.quantity <= total_dice_in_play:
            bid_key = (bid.quantity, bid.face)
            if bid_key not in seen:
                seen.add(bid_key)
                valid_bids.append(bid)
    
    return valid_bids


def apply_pre_reveal_analysis(
    initial_expected_count: float,
    current_bid: Optional[Bid],
    is_special_round: bool,
    total_dice_in_play: int,
    bot_dice: List[int],
    last_bidder_dice: List[int],
    analysis: PlayerAnalysis,
    game_stage: GameStage,
    bot_affinity: float = 1.0,
) -> float:
    """
    Apply analysis of a player's pre-reveal tendencies to adjust the expected count of a bid.
    
    This is "Ability #2".
    
    Args:
        initial_expected_count: Initial expected count calculation
        current_bid: Current bid being analyzed
        is_special_round: Whether special round is active
        total_dice_in_play: Total dice in game
        bot_dice: Bot's dice
        last_bidder_dice: Last bidder's dice (for exclusion from calculation)
        analysis: Player analysis data
        game_stage: Current game stage
        bot_affinity: Bot's affinity for pre-reveal analysis (default 1.0)
    
    Returns:
        Adjusted expected count
    """
    # We need a reasonable amount of data to make a judgment.
    if analysis.pre_reveal_tendency.total < 3:
        return initial_expected_count
    
    if current_bid is None:
        return initial_expected_count
    
    pre_reveal = analysis.pre_reveal_tendency
    bluff_ratio = pre_reveal.bluff_count / pre_reveal.total
    strong_hand_ratio = pre_reveal.strong_hand_count / pre_reveal.total
    
    # To avoid being predictable, the analysis is capped. The cap is lower in early game.
    max_effectiveness = 0.8  # Default cap for TENSE, KNIFE_FIGHT, DUEL
    if game_stage == GameStage.CHAOS:
        max_effectiveness = 0.6
    elif game_stage == GameStage.POSITIVE:
        max_effectiveness = 0.7
    
    # --- Scenario 1: Handle "Known Bluffers" ---
    bluff_threshold = 0.40
    high_bluff_threshold = 0.70
    
    if bluff_ratio > bluff_threshold:
        # Calculate the "pessimistic" scenario where the bidder has NO relevant dice.
        # In special round or when bidding on 1s, they are not wild - count only exact matches.
        # Otherwise, count both the face and 1s (wilds).
        if is_special_round or current_bid.face == 1:
            bot_hand_count = sum(1 for d in bot_dice if d == current_bid.face)
        else:
            bot_hand_count = sum(1 for d in bot_dice if d == current_bid.face or d == 1)
        dice_excluding_bot_and_bidder = total_dice_in_play - len(bot_dice) - len(last_bidder_dice)
        probability = (1.0 / 6.0) if (is_special_round or current_bid.face == 1) else (1.0 / 3.0)
        expected_from_others_excluding_bidder = max(0, dice_excluding_bot_and_bidder) * probability
        pessimistic_expected_count = bot_hand_count + expected_from_others_excluding_bidder
        
        doubt_factor = 0.0
        if bluff_ratio >= high_bluff_threshold:
            # Player is a confirmed liar, we fully distrust them.
            doubt_factor = 1.0
        else:
            # Player is a suspected bluffer. We scale our doubt based on how far they are into the "unknown" zone.
            doubt_factor = (bluff_ratio - bluff_threshold) / (high_bluff_threshold - bluff_threshold)
        
        doubt_factor *= bot_affinity  # The bot's personality affects how much it trusts this analysis.
        doubt_factor = max(0.0, min(max_effectiveness, doubt_factor))  # Clamp with the dynamic cap.
        
        # The final expected count is a weighted average between the normal and pessimistic scenarios.
        return (initial_expected_count * (1 - doubt_factor)) + (pessimistic_expected_count * doubt_factor)
    
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
    Get adjustment factor based on first bid bluff analysis.
    
    Args:
        analysis: Player analysis data
    
    Returns:
        Adjustment factor (0 if not enough data)
    """
    if not analysis or analysis.first_bid_bluffs.total < 2:
        return 0.0  # Not enough data
    
    ratio = analysis.first_bid_bluffs.count / analysis.first_bid_bluffs.total
    if ratio > 0.70:
        return 1.5  # High mistrust -> High adjustment
    if ratio > 0.45:
        return 0.7  # Mild mistrust -> Mild adjustment
    return 0.0


def get_face_pattern_adjustment(analysis: Optional[PlayerAnalysis], bid_face: int) -> float:
    """
    Get adjustment factor based on face bluff pattern.
    
    Args:
        analysis: Player analysis data
        bid_face: Dice face being bid on
    
    Returns:
        Adjustment factor (0 if no pattern detected)
    """
    if not analysis:
        return 0.0
    
    face_key = str(bid_face)
    pattern = analysis.face_bluff_patterns.get(face_key)
    if not pattern or pattern.total_bids < 1:
        return 0.0
    
    ratio = pattern.bluff_count / pattern.total_bids
    if ratio > 0.75:
        return 2.0  # Very strong signal
    if ratio > 0.50:
        return 1.0  # Strong signal
    return 0.0

