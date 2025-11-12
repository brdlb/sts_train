import type { GameState, Player, Bid, DiceValue, GameStage, PlayerAnalysis } from '../../types';

// --- Helper Functions for Local AI Logic ---

export const formatBidFace = (face: DiceValue) => face === 1 ? '★' : face;

// Fix: Added missing formatBid helper function to construct a readable string from a bid object.
export const formatBid = (bid: Bid): string => `${bid.quantity} x ${formatBidFace(bid.face)}`;

/**
 * Determines the current stage of the game based on total dice and active players.
 */
export const getGameStage = (totalDiceInPlay: number, activePlayerCount: number): GameStage => {
    if (activePlayerCount === 2) {
        return 'DUEL';
    }
    if (totalDiceInPlay >= 26) {
        return 'CHAOS';
    }
    if (totalDiceInPlay >= 13) {
        return 'POSITIVE';
    }
    if (totalDiceInPlay >= 6) {
        return 'TENSE';
    }
    return 'KNIFE_FIGHT'; // 5 or fewer dice
};


/**
 * Calculates the statistically expected number of dice for a given face.
 * Now considers if it's a special round where 1s are not wild.
 */
export const calculateExpectedCount = (face: DiceValue, bot: Player, totalDiceInPlay: number, isSpecialRound: boolean): number => {
    const hand = bot.dice;
    const unknownDiceCount = totalDiceInPlay - hand.length;
    // In a special round, or when bidding on 1s, they are not wild.
    const countInHand = hand.filter(d => (isSpecialRound || face === 1) ? d === face : (d === face || d === 1)).length;
    const expectedFromOthers = unknownDiceCount / ((isSpecialRound || face === 1) ? 6 : 3);
    return countInHand + expectedFromOthers;
};


/**
 * Counts the occurrences of each face in a bot's hand, including wilds.
 */
export const getHandStrength = (bot: Player, isSpecialRound: boolean): { [key in DiceValue]: number } => {
    const counts: { [key in DiceValue]: number } = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0 };
    const wilds = isSpecialRound ? 0 : bot.dice.filter(d => d === 1).length;
    bot.dice.forEach(d => counts[d]++);
    if (!isSpecialRound) {
        for (let i = 2; i <= 6; i++) {
            counts[i as DiceValue] += wilds;
        }
    }
    return counts;
};

/**
 * Generates a list of all valid next bids.
 * Now handles special round rules (no face change).
 */
export const generatePossibleNextBids = (currentBid: Bid, totalDiceInPlay: number, isSpecialRound: boolean): Bid[] => {
    const options: Bid[] = [];
    const { quantity, face } = currentBid;

    // In a special round, only quantity can be increased.
    if (isSpecialRound) {
        for (let q = quantity + 1; q <= totalDiceInPlay; q++) {
             options.push({ quantity: q, face: face });
        }
        return options;
    }

    if (face === 1) {
        // RULE: Switching FROM 1s (Pacos)
        const requiredQuantity = quantity * 2 + 1;
        if (requiredQuantity <= totalDiceInPlay) {
            for (let f = 2; f <= 6; f++) {
                options.push({ quantity: requiredQuantity, face: f as DiceValue });
            }
        }
    } else {
        // RULE: Standard bidding (current bid is not on 1s)
        
        // 1. Increase quantity on the same face
        for (let q = quantity + 1; q <= totalDiceInPlay; q++) {
            options.push({ quantity: q, face: face });
        }
        
        // 2. Change to a higher face with the same quantity
        for (let f = face + 1; f <= 6; f++) {
            options.push({ quantity, face: f as DiceValue });
        }
        
        // 3. Switch TO 1s (Pacos)
        const requiredOnesQuantity = Math.ceil(quantity / 2);
        if (requiredOnesQuantity > 0) {
            // Can bid more than the minimum required
             for (let q = requiredOnesQuantity; q <= totalDiceInPlay; q++) {
                options.push({ quantity: q, face: 1 });
            }
        }
    }

    return [...new Set(options.map(o => JSON.stringify(o)))]
        .map(s => JSON.parse(s))
        .filter(b => b.quantity > 0 && b.quantity <= totalDiceInPlay);
};

// --- Player Analysis Helpers ---

export function getFirstBidAdjustment(analysis: PlayerAnalysis | undefined, gameState: GameState): number {
    // Is the current bidder known for bluffing their opening bids?
    if (!analysis || analysis.firstBidBluffs.total < 2) return 0; // Not enough data
    const ratio = analysis.firstBidBluffs.count / analysis.firstBidBluffs.total;
    if (ratio > 0.70) return 1.5; // High mistrust -> High adjustment
    if (ratio > 0.45) return 0.7; // Mild mistrust -> Mild adjustment
    return 0;
}

/**
 * Applies analysis of a player's pre-reveal tendencies to adjust the expected count of a bid.
 * This is "Ability #2".
 * @returns The adjusted expected count.
 */
export const applyPreRevealAnalysis = (
    initialExpectedCount: number,
    gameState: GameState,
    bot: Player,
    lastBidder: Player,
    analysis: PlayerAnalysis
): number => {
    // We need a reasonable amount of data to make a judgment.
    if (!analysis || analysis.preRevealTendency.total < 3) {
        return initialExpectedCount;
    }

    const { preRevealTendency } = analysis;
    const bluffRatio = preRevealTendency.bluffCount / preRevealTendency.total;
    const strongHandRatio = preRevealTendency.strongHandCount / preRevealTendency.total;
    const affinity = bot.personality?.affinities.preRevealAnalysis ?? 1.0;
    
    const { currentBid, isSpecialRound, players } = gameState;
    if (!currentBid) return initialExpectedCount;

    const totalDiceInPlay = players.reduce((sum, p) => sum + p.dice.length, 0);
    const activePlayerCount = players.filter(p => p.dice.length > 0).length;
    const gameStage = getGameStage(totalDiceInPlay, activePlayerCount);

    // To avoid being predictable, the analysis is capped. The cap is lower in early game.
    let maxEffectiveness = 0.8; // Default cap for TENSE, KNIFE_FIGHT, DUEL
    switch (gameStage) {
        case 'CHAOS': maxEffectiveness = 0.6; break;
        case 'POSITIVE': maxEffectiveness = 0.7; break;
    }

    // --- Scenario 1: Handle "Known Bluffers" ---
    const bluffThreshold = 0.40;
    const highBluffThreshold = 0.70;

    if (bluffRatio > bluffThreshold) {
        // Calculate the "pessimistic" scenario where the bidder has NO relevant dice.
        const botHandCount = bot.dice.filter(d => (isSpecialRound || currentBid.face === 1) ? d === currentBid.face : (d === currentBid.face || d === 1)).length;
        const diceExcludingBotAndBidder = totalDiceInPlay - bot.dice.length - lastBidder.dice.length;
        const probability = (isSpecialRound || currentBid.face === 1) ? (1/6) : (1/3);
        const expectedFromOthersExcludingBidder = Math.max(0, diceExcludingBotAndBidder) * probability;
        const pessimisticExpectedCount = botHandCount + expectedFromOthersExcludingBidder;

        let doubtFactor = 0;
        if (bluffRatio >= highBluffThreshold) {
            // Player is a confirmed liar, we fully distrust them.
            doubtFactor = 1.0;
        } else {
            // Player is a suspected bluffer. We scale our doubt based on how far they are into the "unknown" zone.
            doubtFactor = (bluffRatio - bluffThreshold) / (highBluffThreshold - bluffThreshold);
        }

        doubtFactor *= affinity; // The bot's personality affects how much it trusts this analysis.
        doubtFactor = Math.max(0, Math.min(maxEffectiveness, doubtFactor)); // Clamp with the dynamic cap.

        // The final expected count is a weighted average between the normal and pessimistic scenarios.
        return (initialExpectedCount * (1 - doubtFactor)) + (pessimisticExpectedCount * doubtFactor);
    }

    // --- Scenario 2: Handle "Reliable Players" ---
    const strongHandThreshold = 0.40;
    const highStrongHandThreshold = 0.70;

    if (strongHandRatio > strongHandThreshold) {
        let trustFactor = 0;
        if (strongHandRatio >= highStrongHandThreshold) {
            trustFactor = 1.0;
        } else {
            trustFactor = (strongHandRatio - strongHandThreshold) / (highStrongHandThreshold - strongHandThreshold);
        }
        
        trustFactor *= affinity;
        trustFactor = Math.max(0, Math.min(maxEffectiveness, trustFactor)); // Clamp with the dynamic cap.

        // Add a bonus to the expected count, making the bid seem more plausible.
        const trustBonus = trustFactor * 0.75; // Max bonus of +0.75 to expected count.
        return initialExpectedCount + trustBonus;
    }

    // If neither tendency is strong enough, return the original calculation.
    return initialExpectedCount;
};


export function getFacePatternAdjustment(analysis: PlayerAnalysis | undefined, bidFace: DiceValue): number {
    // Does the current bidder have a habit of bluffing on this specific face?
    if (!analysis) return 0;
    const pattern = analysis.faceBluffPatterns[bidFace];
    if (!pattern || pattern.totalBids < 1) return 0;
    const ratio = pattern.bluffCount / pattern.totalBids;
    if (ratio > 0.75) return 2.0; // Very strong signal
    if (ratio > 0.50) return 1.0; // Strong signal
    return 0;
}