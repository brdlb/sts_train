import React from 'react';
import Modal from './Modal';
import Dice from './Dice';
import { PLAYER_NAMES, PLAYER_COLORS } from '../constants';
import { ExtendedActionHistoryEntry } from '../services/api';

interface DiceRevealModalProps {
  isOpen: boolean;
  onClose: () => void;
  actionEntry: ExtendedActionHistoryEntry | null;
  isSpecialRound?: boolean;
}

const DiceRevealModal: React.FC<DiceRevealModalProps> = ({
  isOpen,
  onClose,
  actionEntry,
  isSpecialRound = false,
}) => {
  if (!isOpen || !actionEntry || !actionEntry.consequences) {
    return null;
  }

  const consequences = actionEntry.consequences;
  const allPlayerDice = consequences.all_player_dice || [];
  const actionType = actionEntry.action_type;
  const bidQuantity = consequences.bid_quantity;
  const bidValue = consequences.bid_value;
  const actualCount = consequences.actual_count;
  const bidderId = consequences.bidder_id;
  const challengerId = actionEntry.player_id;

  // Determine if action was successful
  const isChallenge = actionType === 'challenge';
  const isBelieve = actionType === 'believe';
  const success = isChallenge 
    ? consequences.challenge_success 
    : isBelieve 
    ? consequences.believe_success 
    : null;

  // Get loser information
  const loserId = consequences.loser_id;
  const diceLost = consequences.dice_lost || 0;

  // Build title
  const challengerName = PLAYER_NAMES[challengerId] || `Player ${challengerId}`;
  const bidderName = bidderId !== null && bidderId !== undefined
    ? (PLAYER_NAMES[bidderId] || `Player ${bidderId}`)
    : 'Unknown';
  
  let title = '';
  if (isChallenge) {
    title = `${challengerName} вызвал на проверку!`;
  } else if (isBelieve) {
    title = `${challengerName} поверил ставке!`;
  }

  // Build result description
  let resultText = '';
  if (bidQuantity !== null && bidValue !== null) {
    const bidText = `${bidQuantity}x${bidValue}`;
    
    if (isChallenge) {
      if (success === true) {
        resultText = `Ставка ${bidText} от ${bidderName} была неверной! ${bidderName} потерял ${diceLost} кубик(ов).`;
      } else if (success === false) {
        resultText = `Ставка ${bidText} от ${bidderName} была верной! ${challengerName} потерял ${diceLost} кубик(ов).`;
      }
    } else if (isBelieve) {
      if (success === true) {
        resultText = `Ставка ${bidText} от ${bidderName} была точной! ${challengerName} получил преимущество.`;
      } else if (success === false) {
        resultText = `Ставка ${bidText} от ${bidderName} была неточной! ${challengerName} потерял ${diceLost} кубик(ов).`;
      }
    }
  }

  // Count dice for the bid value
  const countDiceForValue = (dice: number[], value: number): number => {
    if (isSpecialRound || value === 1) {
      // In special round or when bidding on 1s, only count exact matches
      return dice.filter(d => d === value).length;
    } else {
      // Normal round: count both the bid value and 1s (wildcards)
      return dice.filter(d => d === value || d === 1).length;
    }
  };

  // Calculate total count across all players
  let totalCount = 0;
  if (bidValue !== null) {
    allPlayerDice.forEach(playerDice => {
      totalCount += countDiceForValue(playerDice, bidValue);
    });
  }

  return (
    <Modal isOpen={isOpen} title={title} onClose={onClose} maxWidth="4xl">
      <div className="space-y-6 max-h-[70vh] overflow-y-auto">
        {/* Result summary */}
        <div className="bg-gray-700/50 rounded-lg p-4">
          <div className="text-lg font-semibold text-white mb-2">Результат:</div>
          <div className="text-gray-300">{resultText}</div>
          {actualCount !== null && (
            <div className="mt-2 text-xl font-bold text-yellow-400">
              Фактическое количество: {actualCount}
            </div>
          )}
          {bidQuantity !== null && bidValue !== null && (
            <div className="mt-2 text-lg text-gray-400">
              Ставка: {bidQuantity}x{bidValue}
            </div>
          )}
        </div>

        {/* All player dice */}
        <div className="space-y-4">
          <div className="text-lg font-semibold text-white">Кубики всех игроков:</div>
          {allPlayerDice.map((playerDice, playerId) => {
            const playerName = PLAYER_NAMES[playerId] || `Player ${playerId}`;
            const playerColor = PLAYER_COLORS[playerId % PLAYER_COLORS.length];
            const isLoser = loserId !== null && loserId === playerId;
            const countForBid = bidValue !== null ? countDiceForValue(playerDice, bidValue) : 0;
            
            return (
              <div
                key={playerId}
                className={`bg-gray-700/50 rounded-lg p-4 ${
                  isLoser ? 'ring-2 ring-red-500' : ''
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    <h3 className="text-xl font-bold text-white">
                      {playerName} {playerId === 0 && '(Вы)'}
                    </h3>
                    {isLoser && (
                      <span className="text-red-400 font-semibold">
                        (Потерял {diceLost} кубик)
                      </span>
                    )}
                  </div>
                  {bidValue !== null && (
                    <div className="text-sm text-gray-400">
                      Подсчет для {bidValue}: {countForBid}
                    </div>
                  )}
                </div>
                <div className="flex flex-wrap gap-2">
                  {playerDice.map((die, index) => {
                    const shouldHighlight = bidValue !== null && (
                      isSpecialRound || bidValue === 1
                        ? die === bidValue
                        : die === bidValue || die === 1
                    );
                    
                    return (
                      <Dice
                        key={index}
                        value={die as 1 | 2 | 3 | 4 | 5 | 6}
                        size={12}
                        revealed={true}
                        color={playerColor}
                        bidForHighlighting={bidValue !== null ? [bidQuantity || 0, bidValue] : null}
                        isSpecialRound={isSpecialRound}
                        gamePhase="reveal"
                      />
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>

        {/* Total count summary */}
        {bidValue !== null && (
          <div className="bg-yellow-500/20 rounded-lg p-4 border-2 border-yellow-500">
            <div className="text-lg font-semibold text-yellow-300 mb-2">
              Общий подсчет для значения {bidValue}:
            </div>
            <div className="text-3xl font-bold text-yellow-400">
              {totalCount}
            </div>
            {bidQuantity !== null && (
              <div className="mt-2 text-gray-300">
                Ставка была: {bidQuantity}
                {totalCount >= bidQuantity ? (
                  <span className="text-green-400 ml-2">✓ Ставка верна</span>
                ) : (
                  <span className="text-red-400 ml-2">✗ Ставка неверна</span>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </Modal>
  );
};

export default DiceRevealModal;

