import React from 'react';
import Modal from './Modal';
import { PLAYER_NAMES, PLAYER_COLORS } from '../constants';
import { GameState } from '../services/api';

interface GameOverModalProps {
  isOpen: boolean;
  onClose: () => void;
  gameState: GameState | null;
}

const GameOverModal: React.FC<GameOverModalProps> = ({
  isOpen,
  onClose,
  gameState,
}) => {
  if (!isOpen || !gameState || !gameState.game_over) {
    return null;
  }

  const winner = gameState.winner;
  const isHumanWinner = winner === 0;
  const winnerName = winner !== null
    ? (PLAYER_NAMES[winner] || `Player ${winner}`)
    : 'Unknown';

  // Calculate statistics from extended_action_history
  let totalBids = 0;
  let totalChallenges = 0;
  let totalBelieves = 0;
  let humanBids = 0;
  let humanChallenges = 0;
  let humanBelieves = 0;

  if (gameState.extended_action_history) {
    gameState.extended_action_history.forEach(entry => {
      if (entry.action_type === 'bid') {
        totalBids++;
        if (entry.player_id === 0) humanBids++;
      } else if (entry.action_type === 'challenge') {
        totalChallenges++;
        if (entry.player_id === 0) humanChallenges++;
      } else if (entry.action_type === 'believe') {
        totalBelieves++;
        if (entry.player_id === 0) humanBelieves++;
      }
    });
  }

  // Count total actions
  const totalActions = totalBids + totalChallenges + totalBelieves;

  return (
    <Modal isOpen={isOpen} title="Игра окончена!" onClose={onClose} maxWidth="2xl" actionLabel="Вернуться в меню">
      <div className="space-y-6 max-h-[70vh] overflow-y-auto">
        {/* Winner announcement */}
        <div className={`rounded-lg p-6 text-center ${isHumanWinner ? 'bg-green-600/20 border-2 border-green-500' : 'bg-red-600/20 border-2 border-red-500'
          }`}>
          <div className="text-4xl mb-2">
            {isHumanWinner ? '🎉' : '😔'}
          </div>
          <h2 className={`text-3xl font-bold mb-2 ${isHumanWinner ? 'text-green-400' : 'text-red-400'
            }`}>
            {isHumanWinner ? 'Вы победили!' : `Победитель: ${winnerName}`}
          </h2>
          {!isHumanWinner && (
            <p className="text-gray-300 text-lg">
              К сожалению, вы проиграли. Попробуйте еще раз!
            </p>
          )}
        </div>

        {/* Final dice counts */}
        <div className="bg-gray-700/50 rounded-lg p-4">
          <h3 className="text-xl font-semibold text-white mb-4">Финальное состояние кубиков:</h3>
          <div className="space-y-2">
            {gameState.player_dice_count.map((diceCount, playerId) => {
              const playerName = PLAYER_NAMES[playerId] || `Player ${playerId}`;
              const playerColor = PLAYER_COLORS[playerId % PLAYER_COLORS.length];
              const isWinner = playerId === winner;
              const isHuman = playerId === 0;

              return (
                <div
                  key={playerId}
                  className={`flex items-center justify-between p-3 rounded ${isWinner ? 'bg-yellow-500/20 border-2 border-yellow-500' : 'bg-gray-600/50'
                    }`}
                >
                  <div className="flex items-center space-x-3">
                    <div className={`w-4 h-4 rounded-full ${playerColor}`}></div>
                    <span className="text-lg font-semibold text-white">
                      {playerName}
                    </span>
                    {isWinner && (
                      <span className="text-yellow-400 font-bold">👑 Победитель</span>
                    )}
                  </div>
                  <div className="text-xl font-bold text-gray-300">
                    {diceCount} {diceCount === 1 ? 'кубик' : diceCount < 5 ? 'кубика' : 'кубиков'}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Game statistics */}
        <div className="bg-gray-700/50 rounded-lg p-4">
          <h3 className="text-xl font-semibold text-white mb-4">Статистика игры:</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-600/50 rounded p-3">
              <div className="text-sm text-gray-400">Всего ходов</div>
              <div className="text-2xl font-bold text-white">{gameState.turn_number}</div>
            </div>
            <div className="bg-gray-600/50 rounded p-3">
              <div className="text-sm text-gray-400">Всего действий</div>
              <div className="text-2xl font-bold text-white">{totalActions}</div>
            </div>
            <div className="bg-gray-600/50 rounded p-3">
              <div className="text-sm text-gray-400">Ставок всего</div>
              <div className="text-2xl font-bold text-white">{totalBids}</div>
              {humanBids > 0 && (
                <div className="text-sm text-blue-400 mt-1">Ваших: {humanBids}</div>
              )}
            </div>
            <div className="bg-gray-600/50 rounded p-3">
              <div className="text-sm text-gray-400">Вызовов всего</div>
              <div className="text-2xl font-bold text-white">{totalChallenges}</div>
              {humanChallenges > 0 && (
                <div className="text-sm text-blue-400 mt-1">Ваших: {humanChallenges}</div>
              )}
            </div>
            <div className="bg-gray-600/50 rounded p-3">
              <div className="text-sm text-gray-400">Believe всего</div>
              <div className="text-2xl font-bold text-white">{totalBelieves}</div>
              {humanBelieves > 0 && (
                <div className="text-sm text-blue-400 mt-1">Ваших: {humanBelieves}</div>
              )}
            </div>
          </div>
        </div>

        {/* Action breakdown */}
        {gameState.extended_action_history && gameState.extended_action_history.length > 0 && (
          <div className="bg-gray-700/50 rounded-lg p-4">
            <h3 className="text-xl font-semibold text-white mb-3">Распределение действий:</h3>
            <div className="space-y-2">
              {gameState.player_dice_count.map((_, playerId) => {
                const playerName = PLAYER_NAMES[playerId] || `Player ${playerId}`;
                const playerActions = gameState.extended_action_history?.filter(
                  e => e.player_id === playerId
                ) || [];
                const playerBids = playerActions.filter(e => e.action_type === 'bid').length;
                const playerChallenges = playerActions.filter(e => e.action_type === 'challenge').length;
                const playerBelieves = playerActions.filter(e => e.action_type === 'believe').length;

                if (playerActions.length === 0) return null;

                return (
                  <div key={playerId} className="flex items-center justify-between p-2 bg-gray-600/30 rounded">
                    <span className="text-gray-300">
                      {playerName}:
                    </span>
                    <div className="flex space-x-4 text-sm">
                      <span className="text-blue-400">Ставок: {playerBids}</span>
                      <span className="text-orange-400">Вызовов: {playerChallenges}</span>
                      <span className="text-purple-400">Believe: {playerBelieves}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </Modal>
  );
};

export default GameOverModal;




