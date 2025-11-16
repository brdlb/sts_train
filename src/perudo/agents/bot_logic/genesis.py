"""
Genesis logic for Standard Stan bot.
"""

from typing import Tuple, Optional, Dict, List
import random
import numpy as np
from ...game.game_state import GameState
from .utils import (
    get_game_stage,
    calculate_expected_count,
    get_hand_strength,
    generate_possible_next_bids,
    apply_pre_reveal_analysis,
    format_bid,
    format_bid_face,
)
from .player_analysis import PlayerAnalysis


def should_stan_start_special_round(
    bot_dice: List[int],
    total_dice_in_play: int,
) -> bool:
    """
    Logic for Standard Stan to decide if he should start a special round.

    Args:
        bot_dice: Bot's dice
        total_dice_in_play: Total number of dice in play

    Returns:
        True if should start special round
    """
    if total_dice_in_play >= 19:
        return False  # 0% chance if 19 or more dice
    if total_dice_in_play <= 3:
        return True  # 100% chance if 3 or fewer dice
    
    # Linear probability scaling for dice count between 4 and 18.
    # The range of dice where probability changes is 19-3 = 16 values.
    # The probability increases as dice count decreases from 18.
    probability = (19 - total_dice_in_play) / 16
    return random.random() < probability


def _generate_stan_initial_bid(
    bot_dice: List[int],
    total_dice_in_play: int,
    is_special_round: bool,
    active_player_count: int,
) -> Tuple[int, int]:
    """
    Logic for Standard Stan's initial bid.

    Args:
        bot_dice: Bot's dice
        total_dice_in_play: Total number of dice in play
        is_special_round: Whether special round is active
        active_player_count: Number of active players

    Returns:
        Initial bid as (quantity, face)
    """
    # CRITICAL: Check if bot has dice
    if not bot_dice:
        # Bot has no dice - return minimal safe bid
        return (1, 2)
    
    # In a special round, Stan has a simple bluff/truth logic.
    if is_special_round:
        if random.random() < 0.40:  # 40% chance to bluff
            own_face = bot_dice[0]
            possible_bluff_faces = [f for f in range(1, 7) if f != own_face]
            if possible_bluff_faces:
                bluff_face = random.choice(possible_bluff_faces)
                return (1, bluff_face)
        # 60% chance to bid own die
        return (1, bot_dice[0])

    hand_strength = get_hand_strength(bot_dice, is_special_round)
    best_face = 2
    max_count = 0
    for face in range(2, 7):
        if hand_strength[face] > max_count:
            max_count = hand_strength[face]
            best_face = face

    # Default "safe" divisor for a standard bot.
    divisor = 4.5
    base_quantity = max(1, int(total_dice_in_play / divisor))
    return (max(base_quantity, max_count), best_face)


def get_standard_stan_decision(
    game_state: GameState,
    bot_id: int,
    bot_dice: List[int],
    player_analysis: Dict[int, PlayerAnalysis],
    round_bid_history: List[Tuple[int, int, int]],  # (player_id, quantity, face)
) -> Tuple[str, Optional[Tuple[int, int]]]:
    """
    The core decision-making logic for Standard Stan.

    Args:
        game_state: Current game state
        bot_id: Bot's player ID
        bot_dice: Bot's dice
        player_analysis: Dictionary of player analysis by player ID
        round_bid_history: History of bids in current round

    Returns:
        Tuple (decision, bid) where decision is 'BID', 'DUDO', or 'CALZA',
        and bid is (quantity, face) or None
    """
    current_bid = game_state.current_bid
    is_special_round = game_state.special_round_active
    total_dice_in_play = sum(game_state.player_dice_count)
    active_player_count = sum(1 for count in game_state.player_dice_count if count > 0)
    game_stage = get_game_stage(total_dice_in_play, active_player_count)

    # --- Initial Bid Logic ---
    if current_bid is None:
        bid = _generate_stan_initial_bid(
            bot_dice, total_dice_in_play, is_special_round, active_player_count
        )
        return ("BID", bid)

    current_bid_quantity, current_bid_face = current_bid
    expected_count = calculate_expected_count(
        current_bid_face, bot_dice, total_dice_in_play, is_special_round
    )
    expected_count_for_decision = expected_count

    # --- DUDO / CALZA Logic ---
    # Risk Tolerance system
    risk_tolerance = 1.0  # Stan's base risk tolerance

    # Game Stage Adjustments to Risk Tolerance
    if game_stage == "CHAOS":
        risk_tolerance += 0.5  # More tolerant in chaotic early game
    elif game_stage == "TENSE":
        risk_tolerance -= 0.25
    elif game_stage == "KNIFE_FIGHT":
        risk_tolerance -= 0.5
    elif game_stage == "DUEL":
        risk_tolerance -= 0.75  # Less tolerant in high-stakes late game

    # Historical Player Analysis & Reality Adjustment
    last_bidder_id = None
    if round_bid_history:
        last_bidder_id = round_bid_history[-1][0]

    if last_bidder_id is not None and last_bidder_id in player_analysis:
        last_bidder_analysis = player_analysis[last_bidder_id]
        is_first_bid_of_round = len(round_bid_history) == 1

        # --- Analysis Ability #1: First Bid Bluff ---
        if (
            is_first_bid_of_round
            and last_bidder_analysis.first_bid_bluffs.total >= 3
        ):
            bluff_ratio = (
                last_bidder_analysis.first_bid_bluffs.count
                / last_bidder_analysis.first_bid_bluffs.total
            )

            is_late_game = game_stage in ("KNIFE_FIGHT", "DUEL")
            activation_threshold = 0.50 if is_late_game else 0.67
            has_credit_of_trust = bluff_ratio < 0.20

            if (
                not has_credit_of_trust
                and bluff_ratio > activation_threshold
                and game_stage != "CHAOS"
            ):
                if is_special_round or current_bid_face == 1:
                    bot_hand_count = bot_dice.count(current_bid_face)
                    probability = 1.0 / 6
                else:
                    bot_hand_count = bot_dice.count(current_bid_face) + bot_dice.count(1)
                    probability = 1.0 / 3

                last_bidder_dice_count = game_state.player_dice_count[last_bidder_id]
                dice_excluding_bot_and_bidder = max(
                    0,
                    total_dice_in_play
                    - len(bot_dice)
                    - last_bidder_dice_count,
                )
                expected_from_others_excluding_bidder = (
                    dice_excluding_bot_and_bidder * probability
                )
                adjusted_expected_count = (
                    bot_hand_count + expected_from_others_excluding_bidder
                )

                confidence = min(1.0, (bluff_ratio - 0.5) * 2)
                expected_count_for_decision = (
                    expected_count_for_decision * (1 - confidence)
                ) + (adjusted_expected_count * confidence)

        # --- Analysis Ability #2: Pre-Reveal Tendency ---
        # This is applied on every bid, potentially stacking with the first bid analysis.
        last_bidder_dice = game_state.get_player_dice(last_bidder_id)
        expected_count_for_decision = apply_pre_reveal_analysis(
            expected_count_for_decision,
            game_state,
            bot_id,
            bot_dice,
            last_bidder_id,
            last_bidder_dice,
            last_bidder_analysis,
            bot_affinity=1.0,  # Standard Stan has neutral affinity
        )

    # Hand Strength Analysis: If the bot has a strong hand, it should be more tolerant of risk.
    if is_special_round or current_bid_face == 1:
        count_in_hand = bot_dice.count(current_bid_face)
    else:
        count_in_hand = bot_dice.count(current_bid_face) + bot_dice.count(1)

    if current_bid_quantity > 1:
        hand_contribution_ratio = count_in_hand / current_bid_quantity
        if hand_contribution_ratio >= 0.75:
            risk_tolerance += 1.5
        elif hand_contribution_ratio >= 0.5:
            risk_tolerance += 0.75

    bid_risk = current_bid_quantity - expected_count_for_decision

    if count_in_hand < current_bid_quantity and bid_risk > risk_tolerance:
        return ("DUDO", None)

    # --- Bidding Logic ---
    possible_bids = generate_possible_next_bids(
        current_bid, total_dice_in_play, is_special_round
    )
    if not possible_bids:
        return ("DUDO", None)

    scored_bids = []
    for bid in possible_bids:
        bid_quantity, bid_face = bid
        score = 0
        bid_expected = calculate_expected_count(
            bid_face, bot_dice, total_dice_in_play, is_special_round
        )
        natural_count = bot_dice.count(bid_face)
        wild_count = 0 if (is_special_round or bid_face == 1) else bot_dice.count(1)

        score += natural_count * 12 + wild_count * 6  # MEDIUM skill base score

        # --- GENESIS LOGIC (The core of Standard Stan) ---
        bid_margin = bid_quantity - bid_expected
        if bid_margin < 0:
            score += abs(bid_margin) * 4  # Safe Play bonus
        else:
            score -= bid_margin ** 2 * 3  # Bluff penalty

        if bid_face == current_bid_face and bid_quantity == current_bid_quantity + 1:
            score += 18  # "Golden Standard" bonus
        elif bid_quantity == current_bid_quantity and bid_face > current_bid_face:
            score += 10  # "Good Option" bonus

        if bid_face == 1 and current_bid_face != 1:  # Switching to 1s
            required_ones_quantity = (current_bid_quantity + 1) // 2  # ceil
            base_switch_bonus = 5 if wild_count > 0 else -5
            if bid_quantity == required_ones_quantity:
                score += base_switch_bonus
            else:
                score -= (bid_quantity - required_ones_quantity) ** 2 * 4

        # --- STANDARD STAN PERSONALITY OVERRIDE ---
        if bid_face == current_bid_face and bid_quantity == current_bid_quantity + 1:
            score += 5  # Extra bonus for his favorite move
        if bid_margin < 0:
            score += abs(bid_margin) * 2  # Extra bonus for safe play
        if bid_face == 1 and current_bid_face != 1:
            if wild_count > 0:
                score += random.random() * wild_count * 2
            else:
                score -= random.random() * 2

        scored_bids.append((bid, score))

    # Find best bid
    best_bid = possible_bids[0]
    highest_score = float("-inf")
    for bid, score in scored_bids:
        if score > highest_score:
            highest_score = score
            best_bid = bid

    return ("BID", best_bid)

