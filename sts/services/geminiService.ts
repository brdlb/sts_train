import type { GameState, Player, BotDecision } from '../types';
import { BOT_PERSONALITIES } from '../constants';
import { getStandardStanDecision, shouldStanStartSpecialRound } from './botLogic/genesis';
import { getPersonalityDecision, shouldOthersStartSpecialRound } from './botLogic/personalities';

/**
 * Determines the bot's decision by dispatching to the correct logic module based on personality.
 * Standard Stan uses his own "genesis" logic, while others use the personality-driven module.
 * @param gameState The current state of the game.
 * @param bot The bot player making the decision.
 * @returns A BotDecision object.
 */
export const getBotDecision = (gameState: GameState, bot: Player): BotDecision => {
    const personalityName = bot.personality!.name;

    // "Ровный Стэн" is the genesis bot and uses his own clean logic file.
    if (personalityName === BOT_PERSONALITIES.STANDARD_STAN.name) {
        return getStandardStanDecision(gameState, bot);
    }

    // All other bots use the personality-driven logic file.
    return getPersonalityDecision(gameState, bot);
};

/**
 * Determines if a bot should call a special round by dispatching to the correct logic module.
 * @param bot The bot player who might start a special round.
 * @param gameState The current state of the game.
 * @returns A boolean indicating whether to start a special round.
 */
export const shouldBotStartSpecialRound = (bot: Player, gameState: GameState): boolean => {
    const personalityName = bot.personality!.name;

    if (personalityName === BOT_PERSONALITIES.STANDARD_STAN.name) {
        return shouldStanStartSpecialRound(bot, gameState);
    }
    return shouldOthersStartSpecialRound(bot, gameState);
};
