"""
Logic for all bot personalities except Standard Stan.
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
from .bot_personalities import BOT_PERSONALITIES


def should_others_start_special_round(
    personality_name: str,
    total_dice_in_play: int,
) -> bool:
    """
    Decides if a non-Stan bot should call a special round.
    
    Args:
        personality_name: Name of bot personality
        total_dice_in_play: Total dice across all players
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


def generate_personality_initial_bid(
    bot_dice: List[int],
    personality: BotPersonality,
    total_dice_in_play: int,
    is_special_round: bool,
    active_player_count: int,
) -> Bid:
    """
    Generates a smart initial bid based on total dice and personality (for non-Stan bots).
    
    Args:
        bot_dice: Bot's dice
        personality: Bot personality
        total_dice_in_play: Total dice across all players
        is_special_round: Whether special round is active
        active_player_count: Number of active players
    """
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
                return Bid(quantity=1, face=bluff_face)
        return Bid(quantity=1, face=bot_dice[0])
    
    hand_strength = get_hand_strength(bot_dice, is_special_round)
    best_face = 2
    max_count = 0
    for f in range(2, 7):
        if hand_strength[f] > max_count:
            max_count = hand_strength[f]
            best_face = f
    
    if active_player_count == 2:  # Duel Logic for all personalities
        skill = personality.skill_level
        randomizer = random.random()
        strategy = "NORMAL"
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
                return Bid(quantity=1, face=random.choice(weak_faces))
        elif strategy == "SLOW_PLAY_STRONG_HAND":
            return Bid(quantity=1, face=best_face)
    
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
    return Bid(quantity=max(base_quantity, max_count), face=best_face)


def get_personality_decision(
    current_bid: Optional[Bid],
    bot_dice: List[int],
    bot_dice_count: int,
    personality: BotPersonality,
    total_dice_in_play: int,
    active_player_count: int,
    is_special_round: bool,
    game_stage: GameStage,
    round_bid_history: List[Dict],  # List of {bid: Bid, bidder_id: str}
    player_analysis: Dict[str, PlayerAnalysis],
    last_bidder_id: Optional[str],
    last_bidder_dice: List[int],
    all_players_dice: List[List[int]],  # All players' dice for finding opponents
    all_player_ids: List[str],  # All player IDs
) -> BotDecision:
    """
    Main decision logic for all bots except Standard Stan.
    
    Args:
        current_bid: Current bid (None if first bid)
        bot_dice: Bot's dice
        bot_dice_count: Number of dice bot has
        personality: Bot personality
        total_dice_in_play: Total dice across all players
        active_player_count: Number of active players
        is_special_round: Whether special round is active
        game_stage: Current game stage
        round_bid_history: History of bids in current round
        player_analysis: Analysis data for all players
        last_bidder_id: ID of last bidder
        last_bidder_dice: Last bidder's dice
        all_players_dice: All players' dice lists
        all_player_ids: All player IDs
    """
    skill = personality.skill_level
    personality_name = personality.name
    
    # --- Initial Bid Logic ---
    if current_bid is None:
        bid = generate_personality_initial_bid(
            bot_dice, personality, total_dice_in_play, is_special_round, active_player_count
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
    if game_stage == GameStage.CHAOS:
        risk_tolerance += 0.5
    elif game_stage == GameStage.TENSE:
        risk_tolerance -= 0.25
    elif game_stage == GameStage.KNIFE_FIGHT:
        risk_tolerance -= 0.5
    elif game_stage == GameStage.DUEL:
        risk_tolerance -= 0.75
    
    if personality_name == BOT_PERSONALITIES["DESPERATE"].name and bot_dice_count > 2:
        risk_tolerance -= 0.5
    
    if game_stage == GameStage.DUEL:
        # Find opponent
        opponent_dice = None
        for i, player_dice in enumerate(all_players_dice):
            if len(player_dice) > 0 and all_player_ids[i] != last_bidder_id:
                opponent_dice = player_dice
                break
        if opponent_dice is not None:
            dice_advantage = bot_dice_count - len(opponent_dice)
            if dice_advantage > 0:
                risk_tolerance -= 0.25
            elif dice_advantage < 0:
                risk_tolerance += 0.5
    
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
                
                is_late_game = game_stage in (GameStage.KNIFE_FIGHT, GameStage.DUEL)
                activation_threshold = 0.50 if is_late_game else 0.67
                
                if personality_name in calculating_bots and bluff_ratio > activation_threshold:
                    should_use_advanced_analysis = True
            
            if should_use_advanced_analysis and game_stage != GameStage.CHAOS:
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
            bot_affinity=personality.affinities.pre_reveal_analysis,
        )
    
    # --- SUPER ABILITY for HARD bots: Analyze the player before the last one ---
    if personality.skill_level == "HARD" and len(round_bid_history) >= 2:
        second_to_last_bid_data = round_bid_history[-2]
        # Handle both dict format and direct Bid objects
        if isinstance(second_to_last_bid_data, dict):
            second_to_last_bidder_id = second_to_last_bid_data.get("bidder_id")
        else:
            second_to_last_bidder_id = None
        
        if second_to_last_bidder_id and second_to_last_bidder_id in player_analysis:
            second_to_last_bidder_analysis = player_analysis[second_to_last_bidder_id]
            if second_to_last_bidder_analysis.pre_reveal_tendency.total >= 3:
                pre_reveal = second_to_last_bidder_analysis.pre_reveal_tendency
                bluff_ratio = pre_reveal.bluff_count / pre_reveal.total
                strong_hand_ratio = pre_reveal.strong_hand_count / pre_reveal.total
                
                # This advanced skill is half as effective as direct analysis
                two_turn_affinity = personality.affinities.pre_reveal_analysis * 0.5
                
                # If the player two turns ago is a known bluffer, it increases general suspicion.
                if bluff_ratio > 0.5:
                    adjustment = min(0.25, (bluff_ratio - 0.5) * 0.5) * two_turn_affinity
                    expected_count_for_decision -= adjustment
                
                # If they are known to be reliable, it slightly increases general trust.
                if strong_hand_ratio > 0.5:
                    adjustment = min(0.25, (strong_hand_ratio - 0.5) * 0.5) * two_turn_affinity
                    expected_count_for_decision += adjustment
    
    # Count in hand
    if is_special_round or current_bid.face == 1:
        count_in_hand = sum(1 for d in bot_dice if d == current_bid.face)
    else:
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
            f"Ставка {format_bid(current_bid)} слишком рискованна. "
            f"Риск {bid_risk:.1f} > моей терпимости {risk_tolerance:.1f}."
        )
        return BotDecision(
            decision="DUDO", bid=None, thought=thought, dialogue="Не верю!"
        )
    
    # Calza Logic
    if personality_name == BOT_PERSONALITIES["DESPERATE"].name and bot_dice_count <= 2:
        if abs(current_bid.quantity - expected_count) < 0.8 and random.random() < 0.65:
            return BotDecision(
                decision="CALZA",
                bid=None,
                thought="В отчаянии я пойду на все! Ставка выглядит правдоподобной!",
                dialogue="Верю! Точно!",
            )
    
    calza_chance = 0.0
    if bot_dice_count <= 3:
        if game_stage == GameStage.DUEL:
            calza_chance = 0.40 + (total_dice_in_play - 2) * (0.125 - 0.40) / (12 - 2)
        else:
            calza_chance = 0.125
    
    if game_stage == GameStage.KNIFE_FIGHT:
        calza_chance = max(calza_chance, 0.25)
    
    if personality_name in (
        BOT_PERSONALITIES["CALCULATING"].name,
        BOT_PERSONALITIES["UNPREDICTABLE"].name,
        BOT_PERSONALITIES["GAMBLER"].name,
        BOT_PERSONALITIES["STATISTICIAN"].name,
    ):
        expert_chance = (
            (0.40 if skill == "HARD" else 0.30) if game_stage == GameStage.DUEL else 0.20
        )
        calza_chance = max(calza_chance, expert_chance)
    
    if (
        abs(current_bid.quantity - expected_count) < 0.5
        and calza_chance > 0
        and random.random() < calza_chance
    ):
        return BotDecision(
            decision="CALZA",
            bid=None,
            thought="Цифры сходятся. Это может быть точное значение.",
            dialogue="Верю! Это точное число.",
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
            0 if (is_special_round or bid.face == 1) else sum(1 for d in bot_dice if d == 1)
        )
        
        if skill == "EASY":
            score += natural_count * 10 + wild_count * 5
        elif skill == "MEDIUM":
            score += natural_count * 12 + wild_count * 6
        elif skill == "HARD":
            score += natural_count * 15 + wild_count * 7
        
        bid_margin = bid.quantity - bid_expected
        if bid_margin < 0:
            score += abs(bid_margin) * 4
        else:
            score -= (bid_margin ** 2) * 3
        
        if current_bid:
            if bid.face == current_bid.face and bid.quantity == current_bid.quantity + 1:
                score += 18
            elif bid.quantity == current_bid.quantity and bid.face > current_bid.face:
                score += 10
        
        if current_bid and bid.face == 1 and current_bid.face != 1:
            required_ones_quantity = (current_bid.quantity + 1) // 2
            base_switch_bonus = 5 if wild_count > 0 else -5
            if bid.quantity == required_ones_quantity:
                score += base_switch_bonus
            else:
                score -= ((bid.quantity - required_ones_quantity) ** 2) * 4
                score += base_switch_bonus / 2
        
        if total_dice_in_play <= 7:
            if is_special_round or bid.face == 1:
                hits_in_hand = sum(1 for d in bot_dice if d == bid.face)
            else:
                hits_in_hand = sum(1 for d in bot_dice if d == bid.face or d == 1)
            score -= (max(0, bid.quantity - hits_in_hand) ** 2) * 15
        
        if game_stage == GameStage.CHAOS:
            if bid_margin > 1.5:
                score += 15
        elif game_stage == GameStage.TENSE:
            if bid_margin > 0.75:
                score -= (bid_margin ** 2) * 5
        elif game_stage == GameStage.KNIFE_FIGHT:
            if bid_margin > 0.25:
                score -= (bid_margin ** 2) * 10
            if current_bid and bid.face == 1 and current_bid.face != 1:
                score += 10
        elif game_stage == GameStage.DUEL:
            # Find opponent
            opponent_dice = None
            for i, player_dice in enumerate(all_players_dice):
                if len(player_dice) > 0 and all_player_ids[i] != last_bidder_id:
                    opponent_dice = player_dice
                    break
            if opponent_dice is not None:
                dice_advantage = bot_dice_count - len(opponent_dice)
                if dice_advantage > 0 and bid_margin > 0.5:
                    score -= (bid_margin ** 2) * (8 if skill == "HARD" else 5)
                elif dice_advantage < 0 and bid_margin > 1 and bid_margin < 3:
                    score += 15 if skill == "HARD" else 10
        
        # --- PERSONALITY OVERRIDES ---
        is_late_game = game_stage in (
            GameStage.TENSE,
            GameStage.KNIFE_FIGHT,
            GameStage.DUEL,
        )
        
        if personality_name in (
            BOT_PERSONALITIES["AGGRESSIVE"].name,
            BOT_PERSONALITIES["LATE_BLOOMER"].name,
        ):
            if personality_name == BOT_PERSONALITIES["LATE_BLOOMER"].name and not is_late_game:
                if bid_margin > 1:
                    score -= 15
            else:
                if bid_margin > 0:
                    score += max(-50, 18 - ((bid_margin - 1.5) ** 2) * 2.5)
        elif personality_name == BOT_PERSONALITIES["DESPERATE"].name:
            if bot_dice_count <= 2:
                if bid_margin > 0:
                    score += max(
                        -50, (15 - ((bid_margin - 2.5) ** 2) * 2) * (6 - bot_dice_count)
                    )
            else:
                if bid_margin > 0 and bid_margin <= 1.5:
                    score += 8
        elif personality_name in (
            BOT_PERSONALITIES["CAUTIOUS"].name,
            BOT_PERSONALITIES["CONSERVATIVE"].name,
        ):
            if bid_margin > (0.25 if skill == "HARD" else 0.5):
                score -= 25
            if (natural_count + wild_count) == 0 and bid.face != 1:
                score -= 25
            if bid.face == 1 and current_bid.face != 1:
                score -= 20
        elif personality_name in (
            BOT_PERSONALITIES["MIMIC"].name,
            BOT_PERSONALITIES["ESCALATOR"].name,
        ):
            if bid.face == current_bid.face:
                score += 15
                jump = bid.quantity - current_bid.quantity
                if jump > 1:
                    score -= (jump ** 2) * 10
            else:
                score -= 30
        elif personality_name == BOT_PERSONALITIES["FOLLOWER"].name:
            if bid.face == current_bid.face:
                score += 10
            else:
                score -= 15
            jump = bid.quantity - current_bid.quantity if current_bid else bid.quantity
            if jump > 1:
                score -= (jump ** 2) * 6
        elif personality_name == BOT_PERSONALITIES["TRAPPER"].name:
            score -= (bid.quantity - current_bid.quantity) * (8 if skill == "HARD" else 6)
            score -= abs(bid.face - current_bid.face) * (8 if skill == "HARD" else 6)
        elif personality_name == BOT_PERSONALITIES["COUNTER"].name:
            # Check if this face was used in round history
            # Handle both dict format {bid: Bid, bidder_id: str} and direct Bid objects
            face_used = False
            for h in round_bid_history:
                if isinstance(h, dict):
                    # Dictionary format: {bid: Bid, bidder_id: str}
                    bid_obj = h.get("bid")
                    if bid_obj and hasattr(bid_obj, "face"):
                        if bid_obj.face == bid.face:
                            face_used = True
                            break
                elif hasattr(h, "face"):
                    # Direct Bid object
                    if h.face == bid.face:
                        face_used = True
                        break
            if face_used:
                score -= 8 if skill == "EASY" else 15
        elif personality_name == BOT_PERSONALITIES["BLUFFER"].name:
            if (natural_count + wild_count) <= 1 and random.random() < 0.80:
                jump = bid.quantity - current_bid.quantity if current_bid else bid.quantity
                bluff_bonus = 0
                if current_bid and bid.face != current_bid.face and jump == 0:
                    bluff_bonus = 35
                elif jump == 1:
                    bluff_bonus = 28
                if jump > 1:
                    bluff_bonus -= (jump ** 2) * 10
                score += (
                    (bluff_bonus * 0.75) if (natural_count + wild_count == 1) else bluff_bonus
                )
        elif personality_name == BOT_PERSONALITIES["GAMBLER"].name:
            if bid.face == 1:
                score += 20
            if current_bid.face != 1 and bid.face == 1:
                score += 15
            if bid_margin > 0.5:
                score += 15
        elif personality_name == BOT_PERSONALITIES["WILDCARD"].name:
            if bid.face == 1:
                ones_count = sum(1 for d in bot_dice if d == 1)
                score += (ones_count * 8) + 5
            else:
                count = sum(1 for d in bot_dice if d == bid.face)
                if count >= 3:
                    score += 18
                elif count >= 2 and wild_count >= 1:
                    score += 10
        elif personality_name == BOT_PERSONALITIES["SABOTEUR"].name:
            if bid_margin > 1.5:
                score += max(-20, 25 - ((bid_margin - 3.5) ** 2) * 2.5)
            if bid.quantity <= bid_expected:
                score -= 10
        elif personality_name == BOT_PERSONALITIES["PROBER"].name:
            if current_bid.face != bid.face:
                score += 8 if skill == "HARD" else 5
            jump = bid.quantity - current_bid.quantity
            if jump > 1:
                score -= jump * 8
        elif personality_name == BOT_PERSONALITIES["STATISTICIAN"].name:
            hand_strength = get_hand_strength(bot_dice, is_special_round)
            best_face = 0
            max_count_val = 0
            for i in range(1, 7):
                if hand_strength[i] > max_count_val:
                    max_count_val = hand_strength[i]
                    best_face = i
            if bid.face == best_face:
                score += 5
        
        scored_bids.append((bid, score))
    
    # Find best bid
    best_bid = possible_bids[0]
    highest_score = float("-inf")
    for bid, score in scored_bids:
        if score > highest_score:
            highest_score = score
            best_bid = bid
    
    # NEMESIS and GRUDGE_HOLDER special logic
    if personality_name in (
        BOT_PERSONALITIES["NEMESIS"].name,
        BOT_PERSONALITIES["GRUDGE_HOLDER"].name,
    ):
        last_bidder = None
        if round_bid_history:
            last_bid_data = round_bid_history[-1]
            # Handle both dict format and direct Bid objects
            if isinstance(last_bid_data, dict):
                last_bidder_id_from_history = last_bid_data.get("bidder_id")
            else:
                last_bidder_id_from_history = None
            if last_bidder_id_from_history:
                # Find last bidder's personality (would need to be passed in)
                # For now, check if last bid matches previous bid face
                is_triggered = False
                if personality_name == BOT_PERSONALITIES["NEMESIS"].name:
                    # Check if last bidder is mimic-like (same face as previous)
                    if len(round_bid_history) >= 2:
                        # Handle both dict format and direct Bid objects
                        last_entry = round_bid_history[-1]
                        prev_entry = round_bid_history[-2]
                        
                        # Extract Bid objects
                        if isinstance(last_entry, dict):
                            last_bid = last_entry.get("bid")
                        elif hasattr(last_entry, "face"):
                            last_bid = last_entry
                        else:
                            last_bid = None
                            
                        if isinstance(prev_entry, dict):
                            prev_bid = prev_entry.get("bid")
                        elif hasattr(prev_entry, "face"):
                            prev_bid = prev_entry
                        else:
                            prev_bid = None
                        
                        if last_bid and prev_bid and hasattr(last_bid, "face") and hasattr(prev_bid, "face"):
                            if last_bid.face == prev_bid.face:
                                is_triggered = True
                elif personality_name == BOT_PERSONALITIES["GRUDGE_HOLDER"].name:
                    is_triggered = True
                
                if is_triggered and current_bid.quantity > (
                    expected_count + (0.25 if skill == "HARD" else 0.75)
                ):
                    return BotDecision(
                        decision="DUDO",
                        bid=None,
                        thought="Что-то здесь не так... Слишком подозрительно.",
                        dialogue="Хватит! Не верю!",
                    )
    
    thought_message = (
        f"Проанализировав варианты, лучшей ставкой кажется {format_bid(best_bid)}."
    )
    dialogue = f"Я ставлю... <strong>{format_bid(best_bid)}</strong>."
    
    return BotDecision(
        decision="BID", bid=best_bid, thought=thought_message, dialogue=dialogue
    )

