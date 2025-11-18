"""
Logic for Standard Stan (genesis bot).
"""

import random
from typing import List, Optional, Dict

from .bot_types import (
    BotPersonality,
    BotDecision,
    Bid,
    GameStage,
    PlayerAnalysis,
)
from .bot_utils import (
    get_game_stage,
    calculate_expected_count,
    get_hand_strength,
    generate_possible_next_bids,
    apply_pre_reveal_analysis,
    format_bid,
)


def should_stan_start_special_round(
    bot_dice: List[int],
    total_dice_in_play: int,
) -> bool:
    """
    Logic for Standard Stan to decide if he should start a special round.
    
    Args:
        bot_dice: Bot's dice
        total_dice_in_play: Total dice across all players
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


def generate_stan_initial_bid(
    bot_dice: List[int],
    total_dice_in_play: int,
    is_special_round: bool,
    active_player_count: int,
) -> Bid:
    """
    Logic for Standard Stan's initial bid.
    
    Args:
        bot_dice: Bot's dice
        total_dice_in_play: Total dice across all players
        is_special_round: Whether special round is active
        active_player_count: Number of active players
    """
    # In a special round, Stan has a simple bluff/truth logic.
    if is_special_round:
        if random.random() < 0.40:  # 40% chance to bluff
            own_face = bot_dice[0]
            possible_bluff_faces = [f for f in range(1, 7) if f != own_face]
            if possible_bluff_faces:
                bluff_face = random.choice(possible_bluff_faces)
                return Bid(quantity=1, face=bluff_face)
        # 60% chance to bid own die
        return Bid(quantity=1, face=bot_dice[0])
    
    hand_strength = get_hand_strength(bot_dice, is_special_round)
    best_face = 2
    max_count = 0
    for f in range(2, 7):
        if hand_strength[f] > max_count:
            max_count = hand_strength[f]
            best_face = f
    
    # Default "safe" divisor for a standard bot.
    divisor = 4.5
    base_quantity = max(1, int(total_dice_in_play / divisor))
    return Bid(quantity=max(base_quantity, max_count), face=best_face)


def get_standard_stan_decision(
    current_bid: Optional[Bid],
    bot_dice: List[int],
    total_dice_in_play: int,
    active_player_count: int,
    is_special_round: bool,
    game_stage: GameStage,
    round_bid_history: List[Dict],  # List of {bid: Bid, bidder_id: str}
    player_analysis: Dict[str, PlayerAnalysis],
    last_bidder_id: Optional[str],
    last_bidder_dice: List[int],
) -> BotDecision:
    """
    The core decision-making logic for Standard Stan.
    
    Args:
        current_bid: Current bid (None if first bid)
        bot_dice: Bot's dice
        total_dice_in_play: Total dice across all players
        active_player_count: Number of active players
        is_special_round: Whether special round is active
        game_stage: Current game stage
        round_bid_history: History of bids in current round
        player_analysis: Analysis data for all players
        last_bidder_id: ID of last bidder
        last_bidder_dice: Last bidder's dice
    """
    # --- Initial Bid Logic ---
    if current_bid is None:
        bid = generate_stan_initial_bid(
            bot_dice, total_dice_in_play, is_special_round, active_player_count
        )
        if is_special_round:
            return BotDecision(
                decision="BID",
                bid=bid,
                thought=f"Это мой Special раунд. Начну со ставки на {format_bid(bid)}.",
                dialogue=f"Играю Special раунд на <strong>{format_bid(bid)}</strong>!",
            )
        return BotDecision(
            decision="BID",
            bid=bid,
            thought="Начну с разумной ставки, основанной на общем количестве костей.",
            dialogue=f"Я думаю, есть как минимум <strong>{format_bid(bid)}</strong>.",
        )
    
    expected_count = calculate_expected_count(
        current_bid.face, bot_dice, total_dice_in_play, is_special_round
    )
    expected_count_for_decision = expected_count
    
    # --- DUDO / CALZA Logic ---
    # NEW: Risk Tolerance system instead of dudoThreshold
    risk_tolerance = 1.0  # Stan's base risk tolerance
    
    # Game Stage Adjustments to Risk Tolerance
    if game_stage == GameStage.CHAOS:
        risk_tolerance += 0.5  # More tolerant in chaotic early game
    elif game_stage == GameStage.TENSE:
        risk_tolerance -= 0.25
    elif game_stage == GameStage.KNIFE_FIGHT:
        risk_tolerance -= 0.5
    elif game_stage == GameStage.DUEL:
        risk_tolerance -= 0.75  # Less tolerant in high-stakes late game
    
    # Historical Player Analysis & Reality Adjustment
    if last_bidder_id and last_bidder_id in player_analysis:
        last_bidder_analysis = player_analysis[last_bidder_id]
        is_first_bid_of_round = len(round_bid_history) == 1
        
        # --- Analysis Ability #1: First Bid Bluff ---
        if is_first_bid_of_round and last_bidder_analysis.first_bid_bluffs.total >= 3:
            bluff_ratio = (
                last_bidder_analysis.first_bid_bluffs.count
                / last_bidder_analysis.first_bid_bluffs.total
            )
            
            is_late_game = game_stage in (GameStage.KNIFE_FIGHT, GameStage.DUEL)
            activation_threshold = 0.50 if is_late_game else 0.67
            has_credit_of_trust = bluff_ratio < 0.20
            
            if (
                not has_credit_of_trust
                and bluff_ratio > activation_threshold
                and game_stage != GameStage.CHAOS
            ):
                if is_special_round or current_bid.face == 1:
                    bot_hand_count = sum(1 for d in bot_dice if d == current_bid.face)
                else:
                    bot_hand_count = sum(1 for d in bot_dice if d == current_bid.face or d == 1)
                dice_excluding_bot_and_bidder = (
                    total_dice_in_play - len(bot_dice) - len(last_bidder_dice)
                )
                probability = (
                    (1.0 / 6.0) if (is_special_round or current_bid.face == 1) else (1.0 / 3.0)
                )
                expected_from_others_excluding_bidder = (
                    max(0, dice_excluding_bot_and_bidder) * probability
                )
                adjusted_expected_count = bot_hand_count + expected_from_others_excluding_bidder
                
                confidence = min(1.0, (bluff_ratio - 0.5) * 2)
                expected_count_for_decision = (
                    expected_count_for_decision * (1 - confidence)
                    + adjusted_expected_count * confidence
                )
        
        # --- Analysis Ability #2: Pre-Reveal Tendency ---
        expected_count_for_decision = apply_pre_reveal_analysis(
            expected_count_for_decision,
            current_bid,
            is_special_round,
            total_dice_in_play,
            bot_dice,
            last_bidder_dice,
            last_bidder_analysis,
            game_stage,
            bot_affinity=1.0,  # Stan's affinity
        )
    
    # Hand Strength Analysis: If the bot has a strong hand, it should be more tolerant of risk.
    if is_special_round or current_bid.face == 1:
        count_in_hand = sum(1 for d in bot_dice if d == current_bid.face)
    else:
        # In normal round: count face OR 1 (joker)
        count_in_hand = sum(1 for d in bot_dice if d == current_bid.face or d == 1)
    if current_bid.quantity > 1:
        hand_contribution_ratio = count_in_hand / current_bid.quantity
        if hand_contribution_ratio >= 0.75:
            risk_tolerance += 1.5
        elif hand_contribution_ratio >= 0.5:
            risk_tolerance += 0.75
    
    bid_risk = current_bid.quantity - expected_count_for_decision
    
    if count_in_hand < current_bid.quantity and bid_risk > risk_tolerance:
        thought = (
            f"Ставка {format_bid(current_bid)} кажется слишком рискованной. "
            f"Мое ожидание {expected_count_for_decision:.1f}, а риск {bid_risk:.1f} "
            f"превышает мою терпимость {risk_tolerance:.1f}."
        )
        return BotDecision(
            decision="DUDO", bid=None, thought=thought, dialogue="Не верю!"
        )
    
    # --- Bidding Logic ---
    possible_bids = generate_possible_next_bids(
        current_bid, total_dice_in_play, is_special_round
    )
    if not possible_bids:
        return BotDecision(
            decision="DUDO",
            bid=None,
            thought="Нет доступных ходов, я должен бросить вызов.",
            dialogue="У меня нет выбора. Не верю!",
        )
    
    scored_bids = []
    for bid in possible_bids:
        score = 0
        bid_expected = calculate_expected_count(
            bid.face, bot_dice, total_dice_in_play, is_special_round
        )
        natural_count = sum(1 for d in bot_dice if d == bid.face)
        wild_count = (
            0
            if (is_special_round or bid.face == 1)
            else sum(1 for d in bot_dice if d == 1)
        )
        
        score += natural_count * 12 + wild_count * 6  # MEDIUM skill base score
        
        # --- GENESIS LOGIC (The core of Standard Stan) ---
        bid_margin = bid.quantity - bid_expected
        if bid_margin < 0:
            score += abs(bid_margin) * 4  # Safe Play bonus
        else:
            score -= (bid_margin ** 2) * 3  # Bluff penalty
        
        if bid.face == current_bid.face and bid.quantity == current_bid.quantity + 1:
            score += 18  # "Golden Standard" bonus
        elif bid.quantity == current_bid.quantity and bid.face > current_bid.face:
            score += 10  # "Good Option" bonus
        
        if bid.face == 1 and current_bid.face != 1:  # Switching to 1s
            required_ones_quantity = (current_bid.quantity + 1) // 2
            base_switch_bonus = 5 if wild_count > 0 else -5
            if bid.quantity == required_ones_quantity:
                score += base_switch_bonus
            else:
                score -= ((bid.quantity - required_ones_quantity) ** 2) * 4
        
        # --- STANDARD STAN PERSONALITY OVERRIDE ---
        if bid.face == current_bid.face and bid.quantity == current_bid.quantity + 1:
            score += 5  # Extra bonus for his favorite move
        if bid_margin < 0:
            score += abs(bid_margin) * 2  # Extra bonus for safe play
        if bid.face == 1 and current_bid.face != 1:
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
    
    thought_message = (
        f"Проанализировав варианты, лучшей ставкой кажется {format_bid(best_bid)}. "
        "Моя уверенность в этой ставке основана на моих костях и вероятностях."
    )
    dialogue = f"Я ставлю... <strong>{format_bid(best_bid)}</strong>."
    
    return BotDecision(
        decision="BID", bid=best_bid, thought=thought_message, dialogue=dialogue
    )

