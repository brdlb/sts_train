import type { GameState, Player, BotDecision, Bid, DiceValue } from '../../types';
import * as utils from './utils';

/**
 * Logic for Standard Stan to decide if he should start a special round.
 */
export const shouldStanStartSpecialRound = (bot: Player, gameState: GameState): boolean => {
    const totalDiceInPlay = gameState.players.reduce((sum, p) => sum + p.dice.length, 0);

    if (totalDiceInPlay >= 19) {
        return false; // 0% chance if 19 or more dice
    }
    if (totalDiceInPlay <= 3) {
        return true; // 100% chance if 3 or fewer dice
    }
    // Linear probability scaling for dice count between 4 and 18.
    // The range of dice where probability changes is 19-3 = 16 values.
    // The probability increases as dice count decreases from 18.
    const probability = (19 - totalDiceInPlay) / 16;
    return Math.random() < probability;
};

/**
 * Logic for Standard Stan's initial bid.
 */
const generateStanInitialBid = (bot: Player, totalDiceInPlay: number, isSpecialRound: boolean, activePlayerCount: number): Bid => {
    // In a special round, Stan has a simple bluff/truth logic.
    if (isSpecialRound) {
        if (Math.random() < 0.40) { // 40% chance to bluff
            const ownFace = bot.dice[0];
            const possibleBluffFaces = ([1, 2, 3, 4, 5, 6] as DiceValue[]).filter(f => f !== ownFace);
            const bluffFace = possibleBluffFaces[Math.floor(Math.random() * possibleBluffFaces.length)];
            return { quantity: 1, face: bluffFace };
        }
        // 60% chance to bid own die
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
    
    // Default "safe" divisor for a standard bot.
    const divisor = 4.5; 
    const baseQuantity = Math.max(1, Math.floor(totalDiceInPlay / divisor));
    return { quantity: Math.max(baseQuantity, maxCount), face: bestFace };
};

/**
 * The core decision-making logic for Standard Stan.
 */
export const getStandardStanDecision = (gameState: GameState, bot: Player): BotDecision => {
    const { currentBid, players, isSpecialRound, playerAnalysis } = gameState;
    const totalDiceInPlay = players.reduce((sum, p) => sum + p.dice.length, 0);
    const activePlayerCount = players.filter(p => p.dice.length > 0).length;
    const gameStage = utils.getGameStage(totalDiceInPlay, activePlayerCount);

    // --- Initial Bid Logic ---
    if (!currentBid) {
        const bid = generateStanInitialBid(bot, totalDiceInPlay, isSpecialRound, activePlayerCount);
        if (isSpecialRound) {
            return {
                decision: 'BID',
                bid,
                thought: `Это мой Special раунд. Начну со ставки на ${utils.formatBidFace(bid.face)}.`,
                dialogue: `Играю Special раунд на <strong>${utils.formatBid(bid)}</strong>!`
            };
        }
        return { decision: 'BID', bid, thought: 'Начну с разумной ставки, основанной на общем количестве костей.', dialogue: `Я думаю, есть как минимум <strong>${utils.formatBid(bid)}</strong>.` };
    }

    const expectedCount = utils.calculateExpectedCount(currentBid.face, bot, totalDiceInPlay, isSpecialRound);
    let expectedCountForDecision = expectedCount;

    // --- DUDO / CALZA Logic ---
    // NEW: Risk Tolerance system instead of dudoThreshold
    let riskTolerance = 1.0; // Stan's base risk tolerance

    // Game Stage Adjustments to Risk Tolerance
    switch (gameStage) {
        case 'CHAOS': riskTolerance += 0.5; break; // More tolerant in chaotic early game
        case 'TENSE': riskTolerance -= 0.25; break;
        case 'KNIFE_FIGHT': riskTolerance -= 0.5; break;
        case 'DUEL': riskTolerance -= 0.75; break; // Less tolerant in high-stakes late game
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
                
                const isLateGame = gameStage === 'KNIFE_FIGHT' || gameStage === 'DUEL';
                const activationThreshold = isLateGame ? 0.50 : 0.67;
                const hasCreditOfTrust = bluffRatio < 0.20;

                if (!hasCreditOfTrust && bluffRatio > activationThreshold && gameStage !== 'CHAOS') {
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

    // Hand Strength Analysis: If the bot has a strong hand, it should be more tolerant of risk.
    const countInHand = bot.dice.filter(d => (isSpecialRound || currentBid.face === 1) ? d === currentBid.face : (d === currentBid.face || d === 1)).length;
    if (currentBid.quantity > 1) {
        const handContributionRatio = countInHand / currentBid.quantity;
        if (handContributionRatio >= 0.75) { riskTolerance += 1.5; }
        else if (handContributionRatio >= 0.5) { riskTolerance += 0.75; }
    }

    const bidRisk = currentBid.quantity - expectedCountForDecision;

    if (countInHand < currentBid.quantity && bidRisk > riskTolerance) {
        const thought = `Ставка ${utils.formatBid(currentBid)} кажется слишком рискованной. Мое ожидание ${expectedCountForDecision.toFixed(1)}, а риск ${bidRisk.toFixed(1)} превышает мою терпимость ${riskTolerance.toFixed(1)}.`;
        return { decision: 'DUDO', bid: null, thought, dialogue: 'Не верю!' };
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
        
        score += naturalCount * 12 + wildCount * 6; // MEDIUM skill base score

        // --- GENESIS LOGIC (The core of Standard Stan) ---
        const bidMargin = bid.quantity - bidExpected;
        if (bidMargin < 0) score += Math.abs(bidMargin) * 4; // Safe Play bonus
        else score -= Math.pow(bidMargin, 2) * 3; // Bluff penalty

        if (bid.face === currentBid.face && bid.quantity === currentBid.quantity + 1) score += 18; // "Golden Standard" bonus
        else if (bid.quantity === currentBid.quantity && bid.face > currentBid.face) score += 10; // "Good Option" bonus

        if (bid.face === 1 && currentBid.face !== 1) { // Switching to 1s
            const requiredOnesQuantity = Math.ceil(currentBid.quantity / 2);
            let baseSwitchBonus = (wildCount > 0) ? 5 : -5;
            if (bid.quantity === requiredOnesQuantity) score += baseSwitchBonus;
            else score -= Math.pow(bid.quantity - requiredOnesQuantity, 2) * 4;
        }

        // --- STANDARD STAN PERSONALITY OVERRIDE ---
        if (bid.face === currentBid.face && bid.quantity === currentBid.quantity + 1) score += 5; // Extra bonus for his favorite move
        if (bidMargin < 0) score += Math.abs(bidMargin) * 2; // Extra bonus for safe play
        if (bid.face === 1 && currentBid.face !== 1) {
            if (wildCount > 0) score += Math.random() * wildCount * 2;
            else score -= Math.random() * 2;
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
    
    const thoughtMessage = `Проанализировав варианты, лучшей ставкой кажется ${utils.formatBid(bestBid)}. Моя уверенность в этой ставке основана на моих костях и вероятностях.`;
    const dialogue = `Я ставлю... <strong>${utils.formatBid(bestBid)}</strong>.`;
    
    return { decision: 'BID', bid: bestBid, thought: thoughtMessage, dialogue };
};