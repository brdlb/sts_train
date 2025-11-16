"""
Personality-based decision logic for all bots except Standard Stan.
"""

from typing import Tuple, Optional, Dict, List, Literal
import random
import numpy as np
from ...game.game_state import GameState
from .constants import BOT_PERSONALITIES, BotPersonality
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


def should_others_start_special_round(
    personality_name: str,
    total_dice_in_play: int,
) -> bool:
    """
    Decides if a non-Stan bot should call a special round.

    Args:
        personality_name: Bot personality name
        total_dice_in_play: Total number of dice in play

    Returns:
        True if should start special round
    """
    probability = 0.02 + max(0, (16 - total_dice_in_play) * 0.065)
    multiplier = 1.0

    if personality_name in (
        BOT_PERSONALITIES["AGGRESSIVE"].name,
        BOT_PERSONALITIES["GAMBLER"].name,
        BOT_PERSONALITIES["LATE_BLOOMER"].name,
    ):
        multiplier = 1.5
    elif personality_name in (
        BOT_PERSONALITIES["CAUTIOUS"].name,
        BOT_PERSONALITIES["CONSERVATIVE"].name,
    ):
        multiplier = 0.4
    elif personality_name in (
        BOT_PERSONALITIES["CALCULATING"].name,
        BOT_PERSONALITIES["STATISTICIAN"].name,
    ):
        multiplier = 1.2 if total_dice_in_play < 10 else 0.6

    final_chance = max(0, min(1.0, probability * multiplier))
    return random.random() < final_chance


def _generate_personality_initial_bid(
    bot_dice: List[int],
    personality: BotPersonality,
    total_dice_in_play: int,
    is_special_round: bool,
    active_player_count: int,
) -> Tuple[int, int]:
    """
    Generates a smart initial bid based on total dice and personality.

    Args:
        bot_dice: Bot's dice
        personality: Bot personality
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
    
    if is_special_round:
        bluff_chance = 0.45
        personality_name = personality.name
        if personality_name in (
            BOT_PERSONALITIES["AGGRESSIVE"].name,
            BOT_PERSONALITIES["BLUFFER"].name,
            BOT_PERSONALITIES["SABOTEUR"].name,
        ):
            bluff_chance = 0.80 if personality.skill_level == "HARD" else 0.70
        elif personality_name == BOT_PERSONALITIES["UNPREDICTABLE"].name:
            bluff_chance = 0.60
        elif personality_name == BOT_PERSONALITIES["DESPERATE"].name:
            bluff_chance = 0.50
        elif personality_name in (
            BOT_PERSONALITIES["CAUTIOUS"].name,
            BOT_PERSONALITIES["CONSERVATIVE"].name,
        ):
            bluff_chance = 0.15
        elif personality_name in (
            BOT_PERSONALITIES["CALCULATING"].name,
            BOT_PERSONALITIES["STATISTICIAN"].name,
        ):
            bluff_chance = 0.30

        if random.random() < bluff_chance:
            own_face = bot_dice[0]
            possible_bluff_faces = [f for f in range(1, 7) if f != own_face]
            if possible_bluff_faces:
                bluff_face = random.choice(possible_bluff_faces)
                return (1, bluff_face)
        return (1, bot_dice[0])

    hand_strength = get_hand_strength(bot_dice, is_special_round)
    best_face = 2
    max_count = 0
    for face in range(2, 7):
        if hand_strength[face] > max_count:
            max_count = hand_strength[face]
            best_face = face

    if active_player_count == 2:  # Duel Logic for all personalities
        skill = personality.skill_level
        randomizer = random.random()
        strategy: Literal["PROBING_BLUFF", "SLOW_PLAY_STRONG_HAND", "NORMAL"] = "NORMAL"
        is_strong_hand = max_count >= 3
        if is_strong_hand:
            if skill == "HARD" and randomizer < 0.70:
                strategy = "SLOW_PLAY_STRONG_HAND"
            elif skill == "MEDIUM" and randomizer < 0.50:
                strategy = "SLOW_PLAY_STRONG_HAND"
        else:
            if skill == "HARD" and randomizer < 0.75:
                strategy = "PROBING_BLUFF"
            elif skill == "MEDIUM" and randomizer < 0.60:
                strategy = "PROBING_BLUFF"
            elif skill == "EASY" and randomizer < 0.30:
                strategy = "PROBING_BLUFF"

        if strategy == "PROBING_BLUFF":
            weak_faces = [f for f in range(2, 7) if hand_strength[f] <= 1]
            if weak_faces:
                return (1, random.choice(weak_faces))
        elif strategy == "SLOW_PLAY_STRONG_HAND":
            return (1, best_face)

    divisor = 4.5
    personality_name = personality.name
    if personality_name in (
        BOT_PERSONALITIES["AGGRESSIVE"].name,
        BOT_PERSONALITIES["BLUFFER"].name,
    ):
        divisor = 3.5
    elif personality_name in (
        BOT_PERSONALITIES["CAUTIOUS"].name,
        BOT_PERSONALITIES["TRAPPER"].name,
        BOT_PERSONALITIES["PROBER"].name,
        BOT_PERSONALITIES["CONSERVATIVE"].name,
    ):
        divisor = 6
    elif personality_name == BOT_PERSONALITIES["UNPREDICTABLE"].name:
        divisor = random.random() * 3 + 3

    base_quantity = max(1, int(total_dice_in_play / divisor))
    return (max(base_quantity, max_count), best_face)


def get_personality_decision(
    game_state: GameState,
    bot_id: int,
    bot_dice: List[int],
    personality: BotPersonality,
    player_analysis: Dict[int, PlayerAnalysis],
    round_bid_history: List[Tuple[int, int, int]],  # (player_id, quantity, face)
) -> Tuple[str, Optional[Tuple[int, int]]]:
    """
    Main decision logic for all bots except Standard Stan.

    Args:
        game_state: Current game state
        bot_id: Bot's player ID
        bot_dice: Bot's dice
        personality: Bot personality
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
    skill = personality.skill_level
    personality_name = personality.name

    # --- Initial Bid Logic ---
    if current_bid is None:
        bid = _generate_personality_initial_bid(
            bot_dice, personality, total_dice_in_play, is_special_round, active_player_count
        )
        return ("BID", bid)

    current_bid_quantity, current_bid_face = current_bid
    expected_count = calculate_expected_count(
        current_bid_face, bot_dice, total_dice_in_play, is_special_round
    )
    expected_count_for_decision = expected_count

    # --- DUDO / CALZA Logic ---
    # Risk Tolerance system
    risk_tolerance = 1.0  # Base for most bots

    # Personality base risk tolerance
    if personality_name in (
        BOT_PERSONALITIES["AGGRESSIVE"].name,
        BOT_PERSONALITIES["LATE_BLOOMER"].name,
        BOT_PERSONALITIES["DESPERATE"].name,
    ):
        risk_tolerance = 1.75
    elif personality_name in (
        BOT_PERSONALITIES["GAMBLER"].name,
        BOT_PERSONALITIES["BLUFFER"].name,
    ):
        risk_tolerance = 1.5
    elif personality_name in (
        BOT_PERSONALITIES["CAUTIOUS"].name,
        BOT_PERSONALITIES["TRAPPER"].name,
        BOT_PERSONALITIES["CONSERVATIVE"].name,
    ):
        risk_tolerance = 0.5
    elif personality_name in (
        BOT_PERSONALITIES["CALCULATING"].name,
        BOT_PERSONALITIES["STATISTICIAN"].name,
    ):
        risk_tolerance = 0.75

    # Game Stage Adjustments to Risk Tolerance
    if game_stage == "CHAOS":
        risk_tolerance += 0.5
    elif game_stage == "TENSE":
        risk_tolerance -= 0.25
    elif game_stage == "KNIFE_FIGHT":
        risk_tolerance -= 0.5
    elif game_stage == "DUEL":
        risk_tolerance -= 0.75

    if personality_name == BOT_PERSONALITIES["DESPERATE"].name and len(bot_dice) > 2:
        risk_tolerance -= 0.5

    if game_stage == "DUEL":
        # Find opponent
        for opp_id in range(game_state.num_players):
            if opp_id != bot_id and game_state.player_dice_count[opp_id] > 0:
                dice_advantage = len(bot_dice) - game_state.player_dice_count[opp_id]
                if dice_advantage > 0:
                    risk_tolerance -= 0.25
                elif dice_advantage < 0:
                    risk_tolerance += 0.5
                break

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
            has_credit_of_trust = bluff_ratio < 0.20
            should_use_advanced_analysis = False

            if not has_credit_of_trust:
                calculating_bots = [
                    BOT_PERSONALITIES["CALCULATING"].name,
                    BOT_PERSONALITIES["STATISTICIAN"].name,
                    BOT_PERSONALITIES["TRAPPER"].name,
                    BOT_PERSONALITIES["COUNTER"].name,
                    BOT_PERSONALITIES["LATE_BLOOMER"].name,
                ]

                is_late_game = game_stage in ("KNIFE_FIGHT", "DUEL")
                activation_threshold = 0.50 if is_late_game else 0.67

                if (
                    personality_name in calculating_bots
                    and bluff_ratio > activation_threshold
                ):
                    should_use_advanced_analysis = True

            if should_use_advanced_analysis and game_stage != "CHAOS":
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
        last_bidder_dice = game_state.get_player_dice(last_bidder_id)
        expected_count_for_decision = apply_pre_reveal_analysis(
            expected_count_for_decision,
            game_state,
            bot_id,
            bot_dice,
            last_bidder_id,
            last_bidder_dice,
            last_bidder_analysis,
            bot_affinity=personality.affinities.pre_reveal_analysis,
        )

    # --- SUPER ABILITY for HARD bots: Analyze the player before the last one ---
    if personality.skill_level == "HARD" and len(round_bid_history) >= 2:
        second_to_last_bid_data = round_bid_history[-2]
        second_to_last_bidder_id = second_to_last_bid_data[0]

        if (
            second_to_last_bidder_id != bot_id
            and second_to_last_bidder_id in player_analysis
        ):
            analysis = player_analysis[second_to_last_bidder_id]
            if analysis.pre_reveal_tendency.total >= 3:
                pre_reveal = analysis.pre_reveal_tendency
                bluff_ratio = (
                    pre_reveal.bluff_count / pre_reveal.total
                    if pre_reveal.total > 0
                    else 0.0
                )
                strong_hand_ratio = (
                    pre_reveal.strong_hand_count / pre_reveal.total
                    if pre_reveal.total > 0
                    else 0.0
                )

                # This advanced skill is half as effective as direct analysis
                two_turn_affinity = personality.affinities.pre_reveal_analysis * 0.5

                # If the player two turns ago is a known bluffer, it increases general suspicion.
                if bluff_ratio > 0.5:
                    adjustment = min(0.25, (bluff_ratio - 0.5) * 0.5) * two_turn_affinity
                    expected_count_for_decision -= adjustment

                # If they are known to be reliable, it slightly increases general trust.
                if strong_hand_ratio > 0.5:
                    adjustment = (
                        min(0.25, (strong_hand_ratio - 0.5) * 0.5) * two_turn_affinity
                    )
                    expected_count_for_decision += adjustment

    # Count in hand
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

    # Calza Logic
    if personality_name == BOT_PERSONALITIES["DESPERATE"].name and len(bot_dice) <= 2:
        if abs(current_bid_quantity - expected_count) < 0.8 and random.random() < 0.65:
            return ("CALZA", None)

    calza_chance = 0.0
    if len(bot_dice) <= 3:
        if game_stage == "DUEL":
            calza_chance = 0.40 + (total_dice_in_play - 2) * (0.125 - 0.40) / (12 - 2)
        else:
            calza_chance = 0.125

    if game_stage == "KNIFE_FIGHT":
        calza_chance = max(calza_chance, 0.25)

    if personality_name in (
        BOT_PERSONALITIES["CALCULATING"].name,
        BOT_PERSONALITIES["UNPREDICTABLE"].name,
        BOT_PERSONALITIES["GAMBLER"].name,
        BOT_PERSONALITIES["STATISTICIAN"].name,
    ):
        if game_stage == "DUEL":
            expert_chance = 0.40 if skill == "HARD" else 0.30
        else:
            expert_chance = 0.20
        calza_chance = max(calza_chance, expert_chance)

    if (
        abs(current_bid_quantity - expected_count) < 0.5
        and calza_chance > 0
        and random.random() < calza_chance
    ):
        return ("CALZA", None)

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

        # Base score by skill level
        if skill == "EASY":
            score += natural_count * 10 + wild_count * 5
        elif skill == "MEDIUM":
            score += natural_count * 12 + wild_count * 6
        else:  # HARD
            score += natural_count * 15 + wild_count * 7

        bid_margin = bid_quantity - bid_expected
        if bid_margin < 0:
            score += abs(bid_margin) * 4
        else:
            score -= bid_margin ** 2 * 3

        if bid_face == current_bid_face and bid_quantity == current_bid_quantity + 1:
            score += 18
        elif bid_quantity == current_bid_quantity and bid_face > current_bid_face:
            score += 10

        if bid_face == 1 and current_bid_face != 1:
            required_ones_quantity = (current_bid_quantity + 1) // 2  # ceil
            base_switch_bonus = 5 if wild_count > 0 else -5
            if bid_quantity == required_ones_quantity:
                score += base_switch_bonus
            else:
                score -= (bid_quantity - required_ones_quantity) ** 2 * 4
                score += base_switch_bonus / 2

        if total_dice_in_play <= 7:
            if is_special_round or bid_face == 1:
                hits_in_hand = bot_dice.count(bid_face)
            else:
                hits_in_hand = bot_dice.count(bid_face) + bot_dice.count(1)
            score -= max(0, bid_quantity - hits_in_hand) ** 2 * 15

        # Game stage adjustments
        if game_stage == "CHAOS":
            if bid_margin > 1.5:
                score += 15
        elif game_stage == "TENSE":
            if bid_margin > 0.75:
                score -= bid_margin ** 2 * 5
        elif game_stage == "KNIFE_FIGHT":
            if bid_margin > 0.25:
                score -= bid_margin ** 2 * 10
            if bid_face == 1 and current_bid_face != 1:
                score += 10
        elif game_stage == "DUEL":
            # Find opponent
            for opp_id in range(game_state.num_players):
                if opp_id != bot_id and game_state.player_dice_count[opp_id] > 0:
                    dice_advantage = len(bot_dice) - game_state.player_dice_count[opp_id]
                    if dice_advantage > 0 and bid_margin > 0.5:
                        penalty_mult = 8 if skill == "HARD" else 5
                        score -= bid_margin ** 2 * penalty_mult
                    elif dice_advantage < 0 and 1 < bid_margin < 3:
                        bonus = 15 if skill == "HARD" else 10
                        score += bonus
                    break

        # --- PERSONALITY OVERRIDES ---
        is_late_game = game_stage in ("TENSE", "KNIFE_FIGHT", "DUEL")

        if personality_name in (
            BOT_PERSONALITIES["AGGRESSIVE"].name,
            BOT_PERSONALITIES["LATE_BLOOMER"].name,
        ):
            if personality_name == BOT_PERSONALITIES["LATE_BLOOMER"].name and not is_late_game:
                if bid_margin > 1:
                    score -= 15
            else:
                if bid_margin > 0:
                    score += max(-50, 18 - (bid_margin - 1.5) ** 2 * 2.5)

        elif personality_name == BOT_PERSONALITIES["DESPERATE"].name:
            if len(bot_dice) <= 2:
                if bid_margin > 0:
                    score += max(-50, (15 - (bid_margin - 2.5) ** 2 * 2) * (6 - len(bot_dice)))
            else:
                if bid_margin > 0 and bid_margin <= 1.5:
                    score += 8

        elif personality_name in (
            BOT_PERSONALITIES["CAUTIOUS"].name,
            BOT_PERSONALITIES["CONSERVATIVE"].name,
        ):
            threshold = 0.25 if skill == "HARD" else 0.5
            if bid_margin > threshold:
                score -= 25
            if (natural_count + wild_count) == 0 and bid_face != 1:
                score -= 25
            if bid_face == 1 and current_bid_face != 1:
                score -= 20

        elif personality_name in (
            BOT_PERSONALITIES["MIMIC"].name,
            BOT_PERSONALITIES["ESCALATOR"].name,
        ):
            if bid_face == current_bid_face:
                score += 15
                if bid_quantity - current_bid_quantity > 1:
                    score -= (bid_quantity - current_bid_quantity) ** 2 * 10
            else:
                score -= 30

        elif personality_name == BOT_PERSONALITIES["FOLLOWER"].name:
            if bid_face == current_bid_face:
                score += 10
            else:
                score -= 15
            jump = bid_quantity - current_bid_quantity
            if jump > 1:
                score -= jump ** 2 * 6

        elif personality_name == BOT_PERSONALITIES["TRAPPER"].name:
            penalty_mult = 8 if skill == "HARD" else 6
            score -= (bid_quantity - current_bid_quantity) * penalty_mult
            score -= abs(bid_face - current_bid_face) * penalty_mult

        elif personality_name == BOT_PERSONALITIES["COUNTER"].name:
            # Check if this face was used in round history
            face_used = any(
                h[2] == bid_face for h in round_bid_history
            )
            if face_used:
                penalty = 8 if skill == "EASY" else 15
                score -= penalty

        elif personality_name == BOT_PERSONALITIES["BLUFFER"].name:
            if (natural_count + wild_count) <= 1 and random.random() < 0.80:
                jump = bid_quantity - current_bid_quantity
                bluff_bonus = 0
                if bid_face != current_bid_face and jump == 0:
                    bluff_bonus = 35
                elif jump == 1:
                    bluff_bonus = 28
                if jump > 1:
                    bluff_bonus -= jump ** 2 * 10
                multiplier = 0.75 if (natural_count + wild_count) == 1 else 1.0
                score += bluff_bonus * multiplier

        elif personality_name == BOT_PERSONALITIES["GAMBLER"].name:
            if bid_face == 1:
                score += 20
            if current_bid_face != 1 and bid_face == 1:
                score += 15
            if bid_margin > 0.5:
                score += 15

        elif personality_name == BOT_PERSONALITIES["WILDCARD"].name:
            if bid_face == 1:
                score += bot_dice.count(1) * 8 + 5
            else:
                count = bot_dice.count(bid_face)
                if count >= 3:
                    score += 18
                elif count >= 2 and wild_count >= 1:
                    score += 10

        elif personality_name == BOT_PERSONALITIES["SABOTEUR"].name:
            if bid_margin > 1.5:
                score += max(-20, 25 - (bid_margin - 3.5) ** 2 * 2.5)
            if bid_quantity <= bid_expected:
                score -= 10

        elif personality_name == BOT_PERSONALITIES["PROBER"].name:
            if current_bid_face != bid_face:
                score += 8 if skill == "HARD" else 5
            if bid_quantity - current_bid_quantity > 1:
                score -= (bid_quantity - current_bid_quantity) * 8

        elif personality_name == BOT_PERSONALITIES["STATISTICIAN"].name:
            hand_strength = get_hand_strength(bot_dice, is_special_round)
            best_face = 0
            max_count = 0
            for face in range(1, 7):
                if hand_strength[face] > max_count:
                    max_count = hand_strength[face]
                    best_face = face
            if bid_face == best_face:
                score += 5

        scored_bids.append((bid, score))

    # Find best bid
    best_bid = possible_bids[0]
    highest_score = float("-inf")
    for bid, score in scored_bids:
        if score > highest_score:
            highest_score = score
            best_bid = bid

    # Special logic for NEMESIS and GRUDGE_HOLDER
    if personality_name in (
        BOT_PERSONALITIES["NEMESIS"].name,
        BOT_PERSONALITIES["GRUDGE_HOLDER"].name,
    ):
        if round_bid_history:
            last_bidder_id = round_bid_history[-1][0]
            is_triggered = False

            if personality_name == BOT_PERSONALITIES["NEMESIS"].name:
                # Check if last bidder is a mimic-type bot (would need personality info)
                # For now, check if same face was used in last two bids
                if len(round_bid_history) >= 2:
                    last_bid = round_bid_history[-1]
                    second_last_bid = round_bid_history[-2]
                    if last_bid[2] == second_last_bid[2]:  # Same face
                        is_triggered = True
            else:  # GRUDGE_HOLDER
                is_triggered = True

            if is_triggered:
                threshold = expected_count + (0.25 if skill == "HARD" else 0.75)
                if current_bid_quantity > threshold:
                    return ("DUDO", None)

    return ("BID", best_bid)

