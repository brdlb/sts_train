import type { GameState, Player, BotDecision, Bid, DiceValue } from '../../types';
import * as utils from './utils';
import { BOT_PERSONALITIES } from '../../constants';

/**
 * Decides if a non-Stan bot should call a special round.
 */
export const shouldOthersStartSpecialRound = (bot: Player, gameState: GameState): boolean => {
    const personalityName = bot.personality?.name;
    const totalDiceInPlay = gameState.players.reduce((sum, p) => sum + p.dice.length, 0);

    let probability = 0.02 + Math.max(0, (16 - totalDiceInPlay) * 0.065);
    let multiplier = 1.0;

    switch(personalityName) {
        case BOT_PERSONALITIES.AGGRESSIVE.name:
        case BOT_PERSONALITIES.GAMBLER.name:
        case BOT_PERSONALITIES.LATE_BLOOMER.name:
             multiplier = 1.5; break;
        case BOT_PERSONALITIES.CAUTIOUS.name:
        case BOT_PERSONALITIES.CONSERVATIVE.name:
             multiplier = 0.4; break;
        case BOT_PERSONALITIES.CALCULATING.name:
        case BOT_PERSONALITIES.STATISTICIAN.name:
             multiplier = totalDiceInPlay < 10 ? 1.2 : 0.6; break;
    }
    
    const finalChance = Math.max(0, Math.min(1.0, probability * multiplier));
    return Math.random() < finalChance;
};

/**
 * Generates a smart initial bid based on total dice and personality (for non-Stan bots).
 */
const generatePersonalityInitialBid = (bot: Player, totalDiceInPlay: number, isSpecialRound: boolean, activePlayerCount: number): Bid => {
    if (isSpecialRound) {
        const personality = bot.personality!;
        let bluffChance = 0.45;
        switch(personality.name) {
            case BOT_PERSONALITIES.AGGRESSIVE.name:
            case BOT_PERSONALITIES.BLUFFER.name:
            case BOT_PERSONALITIES.SABOTEUR.name:
                bluffChance = personality.skillLevel === 'HARD' ? 0.80 : 0.70; break;
            case BOT_PERSONALITIES.UNPREDICTABLE.name: bluffChance = 0.60; break;
            case BOT_PERSONALITIES.DESPERATE.name: bluffChance = 0.50; break;
            case BOT_PERSONALITIES.CAUTIOUS.name:
            case BOT_PERSONALITIES.CONSERVATIVE.name: bluffChance = 0.15; break;
            case BOT_PERSONALITIES.CALCULATING.name:
            case BOT_PERSONALITIES.STATISTICIAN.name: bluffChance = 0.30; break;
        }
        
        if (Math.random() < bluffChance) {
            const ownFace = bot.dice[0];
            const possibleBluffFaces = ([1, 2, 3, 4, 5, 6] as DiceValue[]).filter(f => f !== ownFace);
            if (possibleBluffFaces.length > 0) {
                const bluffFace = possibleBluffFaces[Math.floor(Math.random() * possibleBluffFaces.length)];
                return { quantity: 1, face: bluffFace };
            }
        }
        return { quantity: 1, face: bot.dice[0] };
    }

    const handStrength = utils.getHandStrength(bot, isSpecialRound);
    let bestFace: DiceValue = 2;
    let maxCount = 0;
    for (let f = 2; f <= 6; f++) {
        if (handStrength[f as DiceValue] > maxCount) {
            maxCount = handStrength[f as DiceValue];
            bestFace = f as DiceValue;
        }
    }
    
    if (activePlayerCount === 2) { // Duel Logic for all personalities
        const skill = bot.personality!.skillLevel;
        const randomizer = Math.random();
        let strategy: 'PROBING_BLUFF' | 'SLOW_PLAY_STRONG_HAND' | 'NORMAL' = 'NORMAL';
        const isStrongHand = maxCount >= 3;
        if (isStrongHand) {
            if (skill === 'HARD' && randomizer < 0.70) strategy = 'SLOW_PLAY_STRONG_HAND';
            else if (skill === 'MEDIUM' && randomizer < 0.50) strategy = 'SLOW_PLAY_STRONG_HAND';
        } else {
            if (skill === 'HARD' && randomizer < 0.75) strategy = 'PROBING_BLUFF';
            else if (skill === 'MEDIUM' && randomizer < 0.60) strategy = 'PROBING_BLUFF';
            else if (skill === 'EASY' && randomizer < 0.30) strategy = 'PROBING_BLUFF';
        }
        switch (strategy) {
            case 'PROBING_BLUFF':
                const weakFaces = ([2, 3, 4, 5, 6] as DiceValue[]).filter(f => handStrength[f as DiceValue] <= 1);
                if (weakFaces.length > 0) return { quantity: 1, face: weakFaces[Math.floor(Math.random() * weakFaces.length)] };
                break;
            case 'SLOW_PLAY_STRONG_HAND':
                return { quantity: 1, face: bestFace };
        }
    }

    let divisor = 4.5;
    const personalityName = bot.personality?.name;
    switch (personalityName) {
        case BOT_PERSONALITIES.AGGRESSIVE.name:
        case BOT_PERSONALITIES.BLUFFER.name: divisor = 3.5; break;
        case BOT_PERSONALITIES.CAUTIOUS.name:
        case BOT_PERSONALITIES.TRAPPER.name:
        case BOT_PERSONALITIES.PROBER.name:
        case BOT_PERSONALITIES.CONSERVATIVE.name: divisor = 6; break;
        case BOT_PERSONALITIES.UNPREDICTABLE.name: divisor = Math.random() * 3 + 3; break;
    }

    const baseQuantity = Math.max(1, Math.floor(totalDiceInPlay / divisor));
    return { quantity: Math.max(baseQuantity, maxCount), face: bestFace };
};

/**
 * Main decision logic for all bots except Standard Stan.
 */
export const getPersonalityDecision = (gameState: GameState, bot: Player): BotDecision => {
    const { currentBid, players, isSpecialRound, playerAnalysis } = gameState;
    const totalDiceInPlay = players.reduce((sum, p) => sum + p.dice.length, 0);
    const activePlayerCount = players.filter(p => p.dice.length > 0).length;
    const gameStage = utils.getGameStage(totalDiceInPlay, activePlayerCount);
    const personality = bot.personality!;
    const skill = personality.skillLevel;
    const personalityName = personality.name;

    // --- Initial Bid Logic ---
    if (!currentBid) {
        const bid = generatePersonalityInitialBid(bot, totalDiceInPlay, isSpecialRound, activePlayerCount);
        if (isSpecialRound) {
            return { decision: 'BID', bid, thought: `Это мой Special раунд. Начну со ставки на ${utils.formatBidFace(bid.face)}.`, dialogue: `Играю Special раунд на <strong>${utils.formatBid(bid)}</strong>!` };
        }
        return { decision: 'BID', bid, thought: 'Начну с разумной ставки, основанной на общем количестве костей.', dialogue: `Я думаю, есть как минимум <strong>${utils.formatBid(bid)}</strong>.` };
    }

    const expectedCount = utils.calculateExpectedCount(currentBid.face, bot, totalDiceInPlay, isSpecialRound);
    let expectedCountForDecision = expectedCount;

    // --- DUDO / CALZA Logic ---
    // NEW: Risk Tolerance system instead of dudoThreshold
    let riskTolerance = 1.0; // Base for most bots

    // Personality base risk tolerance
    switch (personalityName) {
        case BOT_PERSONALITIES.AGGRESSIVE.name:
        case BOT_PERSONALITIES.LATE_BLOOMER.name:
        case BOT_PERSONALITIES.DESPERATE.name:
            riskTolerance = 1.75; break;
        case BOT_PERSONALITIES.GAMBLER.name:
        case BOT_PERSONALITIES.BLUFFER.name:
            riskTolerance = 1.5; break;
        case BOT_PERSONALITIES.CAUTIOUS.name:
        case BOT_PERSONALITIES.TRAPPER.name:
        case BOT_PERSONALITIES.CONSERVATIVE.name:
            riskTolerance = 0.5; break;
        case BOT_PERSONALITIES.CALCULATING.name:
        case BOT_PERSONALITIES.STATISTICIAN.name:
            riskTolerance = 0.75; break;
    }

    // Game Stage Adjustments to Risk Tolerance
    switch (gameStage) {
        case 'CHAOS': riskTolerance += 0.5; break;
        case 'TENSE': riskTolerance -= 0.25; break;
        case 'KNIFE_FIGHT': riskTolerance -= 0.5; break;
        case 'DUEL': riskTolerance -= 0.75; break;
    }

    if (personalityName === BOT_PERSONALITIES.DESPERATE.name && bot.dice.length > 2) riskTolerance -= 0.5;
    if (gameStage === 'DUEL') {
        const opponent = players.find(p => p.id !== bot.id && p.dice.length > 0);
        if (opponent) {
            const diceAdvantage = bot.dice.length - opponent.dice.length;
            if (diceAdvantage > 0) riskTolerance -= 0.25;
            else if (diceAdvantage < 0) riskTolerance += 0.5;
        }
    }
    
    // Historical Player Analysis & Reality Adjustment
    const lastBidderIndex = gameState.lastBidderIndex;
    const lastBidder = lastBidderIndex !== null ? players[lastBidderIndex] : null;

    if (lastBidder) {
        const lastBidderAnalysis = playerAnalysis[lastBidder.id];
        if (lastBidderAnalysis) {
            const isFirstBidOfRound = gameState.roundBidHistory.length === 1;

            // --- Analysis Ability #1: First Bid Bluff ---
            if (isFirstBidOfRound && lastBidderAnalysis.firstBidBluffs.total >= 3) {
                const bluffRatio = lastBidderAnalysis.firstBidBluffs.count / lastBidderAnalysis.firstBidBluffs.total;
                const hasCreditOfTrust = bluffRatio < 0.20;
                let shouldUseAdvancedAnalysis = false;
                
                if (!hasCreditOfTrust) {
                    const calculatingBots = [
                        BOT_PERSONALITIES.CALCULATING.name, BOT_PERSONALITIES.STATISTICIAN.name,
                        BOT_PERSONALITIES.TRAPPER.name, BOT_PERSONALITIES.COUNTER.name,
                        BOT_PERSONALITIES.LATE_BLOOMER.name
                    ];
                    
                    const isLateGame = gameStage === 'KNIFE_FIGHT' || gameStage === 'DUEL';
                    const activationThreshold = isLateGame ? 0.50 : 0.67;
                    
                    if (calculatingBots.includes(personalityName) && bluffRatio > activationThreshold) {
                        shouldUseAdvancedAnalysis = true;
                    }
                }
                
                if (shouldUseAdvancedAnalysis && gameStage !== 'CHAOS') {
                    const botHandCount = bot.dice.filter(d => (isSpecialRound || currentBid.face === 1) ? d === currentBid.face : (d === currentBid.face || d === 1)).length;
                    const diceExcludingBotAndBidder = totalDiceInPlay - bot.dice.length - lastBidder.dice.length;
                    const probability = (isSpecialRound || currentBid.face === 1) ? (1/6) : (1/3);
                    const expectedFromOthersExcludingBidder = Math.max(0, diceExcludingBotAndBidder) * probability;
                    const adjustedExpectedCount = botHandCount + expectedFromOthersExcludingBidder;
                    
                    const confidence = Math.min(1.0, (bluffRatio - 0.5) * 2);
                    expectedCountForDecision = (expectedCountForDecision * (1 - confidence)) + (adjustedExpectedCount * confidence);
                }
            }
            
            // --- Analysis Ability #2: Pre-Reveal Tendency ---
            // This is applied on every bid, potentially stacking with the first bid analysis.
            expectedCountForDecision = utils.applyPreRevealAnalysis(
                expectedCountForDecision, // Use the (potentially modified) value
                gameState,
                bot,
                lastBidder,
                lastBidderAnalysis
            );
        }
    }
    
    // --- SUPER ABILITY for HARD bots: Analyze the player before the last one ---
    if (personality.skillLevel === 'HARD' && gameState.roundBidHistory.length >= 2) {
        const secondToLastBidData = gameState.roundBidHistory[gameState.roundBidHistory.length - 2];
        const secondToLastBidder = players.find(p => p.id === secondToLastBidData.bidderId);

        if (secondToLastBidder && secondToLastBidder.id !== bot.id) {
            const analysis = playerAnalysis[secondToLastBidder.id];
            if (analysis && analysis.preRevealTendency.total >= 3) {
                const { total, bluffCount, strongHandCount } = analysis.preRevealTendency;
                const bluffRatio = bluffCount / total;
                const strongHandRatio = strongHandCount / total;
                
                // This advanced skill is half as effective as direct analysis and uses the bot's own affinity.
                const twoTurnAffinity = personality.affinities.preRevealAnalysis * 0.5;

                // If the player two turns ago is a known bluffer, it increases general suspicion.
                if (bluffRatio > 0.5) {
                    const adjustment = Math.min(0.25, (bluffRatio - 0.5) * 0.5) * twoTurnAffinity;
                    expectedCountForDecision -= adjustment;
                }
                
                // If they are known to be reliable, it slightly increases general trust.
                if (strongHandRatio > 0.5) {
                     const adjustment = Math.min(0.25, (strongHandRatio - 0.5) * 0.5) * twoTurnAffinity;
                     expectedCountForDecision += adjustment;
                }
            }
        }
    }


    const countInHand = bot.dice.filter(d => (isSpecialRound || currentBid.face === 1) ? d === currentBid.face : (d === currentBid.face || d === 1)).length;
    if (currentBid.quantity > 1) {
        const handContributionRatio = countInHand / currentBid.quantity;
        if (handContributionRatio >= 0.75) riskTolerance += 1.5;
        else if (handContributionRatio >= 0.5) riskTolerance += 0.75;
    }

    const bidRisk = currentBid.quantity - expectedCountForDecision;

    if (countInHand < currentBid.quantity && bidRisk > riskTolerance) {
        const thought = `Ставка ${utils.formatBid(currentBid)} слишком рискованна. Риск ${bidRisk.toFixed(1)} > моей терпимости ${riskTolerance.toFixed(1)}.`;
        return { decision: 'DUDO', bid: null, thought, dialogue: 'Не верю!' };
    }
    
    // Calza Logic
    if (personalityName === BOT_PERSONALITIES.DESPERATE.name && bot.dice.length <= 2) {
        if (Math.abs(currentBid.quantity - expectedCount) < 0.8 && Math.random() < 0.65) {
            return { decision: 'CALZA', bid: null, thought: 'В отчаянии я пойду на все! Ставка выглядит правдоподобной!', dialogue: 'Верю! Точно!' };
        }
    }

    let calzaChance = 0.0;
    if (bot.dice.length <= 3) calzaChance = (gameStage === 'DUEL') ? (0.40 + (totalDiceInPlay - 2) * (0.125 - 0.40) / (12 - 2)) : 0.125;
    if (gameStage === 'KNIFE_FIGHT') calzaChance = Math.max(calzaChance, 0.25);
    if ([BOT_PERSONALITIES.CALCULATING.name, BOT_PERSONALITIES.UNPREDICTABLE.name, BOT_PERSONALITIES.GAMBLER.name, BOT_PERSONALITIES.STATISTICIAN.name].includes(personalityName)) {
        let expertChance = (gameStage === 'DUEL') ? (skill === 'HARD' ? 0.40 : 0.30) : 0.20;
        calzaChance = Math.max(calzaChance, expertChance);
    }
    if (Math.abs(currentBid.quantity - expectedCount) < 0.5 && calzaChance > 0 && Math.random() < calzaChance) {
         return { decision: 'CALZA', bid: null, thought: 'Цифры сходятся. Это может быть точное значение.', dialogue: 'Верю! Это точное число.' };
    }

    // --- Bidding Logic ---
    const possibleBids = utils.generatePossibleNextBids(currentBid, totalDiceInPlay, isSpecialRound);
    if (possibleBids.length === 0) {
        return { decision: 'DUDO', bid: null, thought: 'Нет доступных ходов, я должен бросить вызов.', dialogue: 'У меня нет выбора. Не верю!' };
    }

    const scoredBids = possibleBids.map(bid => {
        let score = 0;
        const bidExpected = utils.calculateExpectedCount(bid.face, bot, totalDiceInPlay, isSpecialRound);
        const naturalCount = bot.dice.filter(d => d === bid.face).length;
        const wildCount = (isSpecialRound || bid.face === 1) ? 0 : bot.dice.filter(d => d === 1).length;
        
        switch (skill) {
            case 'EASY': score += naturalCount * 10 + wildCount * 5; break;
            case 'MEDIUM': score += naturalCount * 12 + wildCount * 6; break;
            case 'HARD': score += naturalCount * 15 + wildCount * 7; break;
        }

        const bidMargin = bid.quantity - bidExpected;
        if (bidMargin < 0) score += Math.abs(bidMargin) * 4;
        else score -= Math.pow(bidMargin, 2) * 3;

        if (currentBid) {
            if (bid.face === currentBid.face && bid.quantity === currentBid.quantity + 1) score += 18;
            else if (bid.quantity === currentBid.quantity && bid.face > currentBid.face) score += 10;
        }

        if (currentBid && bid.face === 1 && currentBid.face !== 1) {
            const requiredOnesQuantity = Math.ceil(currentBid.quantity / 2);
            let baseSwitchBonus = (wildCount > 0) ? 5 : -5;
            if (bid.quantity === requiredOnesQuantity) score += baseSwitchBonus;
            else {
                score -= Math.pow(bid.quantity - requiredOnesQuantity, 2) * 4;
                score += baseSwitchBonus / 2;
            }
        }
        
        if (totalDiceInPlay <= 7) {
            const hitsInHand = (isSpecialRound || bid.face === 1) ? bot.dice.filter(d => d === bid.face).length : bot.dice.filter(d => d === bid.face || d === 1).length;
            score -= Math.pow(Math.max(0, bid.quantity - hitsInHand), 2) * 15;
        }
        
        switch (gameStage) {
            case 'CHAOS': if (bidMargin > 1.5) score += 15; break;
            case 'TENSE': if (bidMargin > 0.75) score -= Math.pow(bidMargin, 2) * 5; break;
            case 'KNIFE_FIGHT':
                if (bidMargin > 0.25) score -= Math.pow(bidMargin, 2) * 10;
                if (currentBid && bid.face === 1 && currentBid.face !== 1) score += 10;
                break;
            case 'DUEL':
                 const opponent = players.find(p => p.id !== bot.id && p.dice.length > 0);
                 if (opponent) {
                     const diceAdvantage = bot.dice.length - opponent.dice.length;
                     if (diceAdvantage > 0 && bidMargin > 0.5) score -= Math.pow(bidMargin, 2) * (skill === 'HARD' ? 8 : 5);
                     else if (diceAdvantage < 0 && bidMargin > 1 && bidMargin < 3) score += (skill === 'HARD' ? 15 : 10);
                 }
                break;
        }

        // --- PERSONALITY OVERRIDES ---
        const isLateGame = gameStage === 'TENSE' || gameStage === 'KNIFE_FIGHT' || gameStage === 'DUEL';
        switch (personality.name) {
            case BOT_PERSONALITIES.AGGRESSIVE.name:
            case BOT_PERSONALITIES.LATE_BLOOMER.name:
                if (personality.name === BOT_PERSONALITIES.LATE_BLOOMER.name && !isLateGame) { if (bidMargin > 1) score -= 15; break; }
                if (bidMargin > 0) score += Math.max(-50, (18 - Math.pow(bidMargin - 1.5, 2) * 2.5));
                break;
            case BOT_PERSONALITIES.DESPERATE.name:
                if (bot.dice.length <= 2) { if (bidMargin > 0) score += Math.max(-50, (15 - Math.pow(bidMargin - 2.5, 2) * 2) * (6 - bot.dice.length)); }
                else { if (bidMargin > 0 && bidMargin <= 1.5) score += 8; }
                break;
            case BOT_PERSONALITIES.CAUTIOUS.name:
            case BOT_PERSONALITIES.CONSERVATIVE.name:
                if (bidMargin > (skill === 'HARD' ? 0.25 : 0.5)) score -= 25;
                if ((naturalCount + wildCount) === 0 && bid.face !== 1) score -= 25;
                if (bid.face === 1 && currentBid.face !== 1) score -= 20;
                break;
            case BOT_PERSONALITIES.MIMIC.name:
            case BOT_PERSONALITIES.ESCALATOR.name:
                if (bid.face === currentBid.face) {
                    score += 15;
                    if (bid.quantity - currentBid.quantity > 1) score -= Math.pow(bid.quantity - currentBid.quantity, 2) * 10;
                } else score -= 30;
                break;
            case BOT_PERSONALITIES.FOLLOWER.name:
                 if (bid.face === currentBid.face) score += 10; else score -= 15;
                 const jump = currentBid ? bid.quantity - currentBid.quantity : bid.quantity;
                 if (jump > 1) score -= Math.pow(jump, 2) * 6;
                 break;
            case BOT_PERSONALITIES.TRAPPER.name:
                score -= (bid.quantity - currentBid.quantity) * (skill === 'HARD' ? 8 : 6);
                score -= Math.abs(bid.face - currentBid.face) * (skill === 'HARD' ? 8 : 6);
                break;
            case BOT_PERSONALITIES.COUNTER.name:
                if (gameState.roundBidHistory.some(h => h.bid.face === bid.face)) score -= (skill === 'EASY' ? 8 : 15);
                break;
            case BOT_PERSONALITIES.BLUFFER.name:
                if (naturalCount + wildCount <= 1 && Math.random() < 0.80) {
                    const jump = currentBid ? bid.quantity - currentBid.quantity : bid.quantity;
                    let bluffBonus = 0;
                    if (currentBid && bid.face !== currentBid.face && jump === 0) bluffBonus = 35;
                    else if (jump === 1) bluffBonus = 28;
                    if (jump > 1) bluffBonus -= Math.pow(jump, 2) * 10;
                    score += (naturalCount + wildCount === 1) ? bluffBonus * 0.75 : bluffBonus;
                }
                break;
            case BOT_PERSONALITIES.GAMBLER.name:
                 if (bid.face === 1) score += 20;
                 if (currentBid.face !== 1 && bid.face === 1) score += 15;
                 if (bidMargin > 0.5) score += 15;
                 break;
            case BOT_PERSONALITIES.WILDCARD.name:
                 if (bid.face === 1) score += (bot.dice.filter(d => d === 1).length * 8) + 5;
                 else {
                    const count = bot.dice.filter(d => d === bid.face).length;
                    if (count >= 3) score += 18;
                    else if (count >= 2 && wildCount >= 1) score += 10;
                 }
                 break;
            case BOT_PERSONALITIES.SABOTEUR.name:
                 if (bidMargin > 1.5) score += Math.max(-20, 25 - Math.pow(bidMargin - 3.5, 2) * 2.5);
                 if (bid.quantity <= bidExpected) score -= 10;
                 break;
            case BOT_PERSONALITIES.PROBER.name:
                if (currentBid.face !== bid.face) score += (skill === 'HARD' ? 8 : 5);
                if (bid.quantity - currentBid.quantity > 1) score -= ((bid.quantity - currentBid.quantity) * 8);
                break;
            case BOT_PERSONALITIES.STATISTICIAN.name:
                const handStrength = utils.getHandStrength(bot, isSpecialRound);
                let bestFace = 0, maxCount = 0;
                for (let i = 1; i <= 6; i++) {
                    if (handStrength[i as DiceValue] > maxCount) {
                        maxCount = handStrength[i as DiceValue]; bestFace = i;
                    }
                }
                if (bid.face === bestFace) score += 5;
                break;
        }
        return { bid, score };
    });

    let bestBid = possibleBids[0];
    let highestScore = -Infinity;
    for (const scored of scoredBids) {
        if (scored.score > highestScore) {
            highestScore = scored.score;
            bestBid = scored.bid;
        }
    }
    
    if (personality.name === BOT_PERSONALITIES.NEMESIS.name || personality.name === BOT_PERSONALITIES.GRUDGE_HOLDER.name) {
         const history = gameState.roundBidHistory;
         const lastBidder = players.find(p => p.id === history[history.length - 1]?.bidderId);
         let isTriggered = false;
         if (personality.name === BOT_PERSONALITIES.NEMESIS.name) {
            const isMimicry = [BOT_PERSONALITIES.MIMIC.name, BOT_PERSONALITIES.ESCALATOR.name, BOT_PERSONALITIES.FOLLOWER.name].includes(lastBidder?.personality?.name || '');
            isTriggered = isMimicry || (history.length >= 2 && history[history.length - 1].bid.face === history[history.length - 2].bid.face);
         }
         if(personality.name === BOT_PERSONALITIES.GRUDGE_HOLDER.name) isTriggered = true;
         if (isTriggered && currentBid.quantity > (expectedCount + (skill === 'HARD' ? 0.25 : 0.75))) {
             return { decision: 'DUDO', bid: null, thought: `Что-то здесь не так... Слишком подозрительно.`, dialogue: `Хватит! Не верю!` };
         }
    }

    let thoughtMessage = `Проанализировав варианты, лучшей ставкой кажется ${utils.formatBid(bestBid)}.`;
    const dialogue = `Я ставлю... <strong>${utils.formatBid(bestBid)}</strong>.`;
    
    return { decision: 'BID', bid: bestBid, thought: thoughtMessage, dialogue };
};