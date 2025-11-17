import React from 'react';
import { GameHistory } from '../services/api';
import { getPlayerName } from '../utils/playerHelpers';

interface GameHistoryModalProps {
  history: GameHistory | null;
  onClose: () => void;
}

const formatActionDescription = (
  action: GameHistory['actions'][0],
  players: GameHistory['players']
): { description: string; details?: string } => {
  const playerName = getPlayerName(action.player_id);
  const { action_type, action_data } = action;

  if (action_type === 'bid') {
    const quantity = action_data?.quantity;
    const value = action_data?.value;
    if (quantity !== null && quantity !== undefined && value !== null && value !== undefined) {
      return {
        description: `${playerName} сделал ставку: ${quantity}x${value}`,
      };
    }
    return {
      description: `${playerName} сделал ставку`,
    };
  } else if (action_type === 'challenge') {
    return {
      description: `${playerName} оспорил ставку`,
    };
  } else if (action_type === 'believe') {
    return {
      description: `${playerName} поверил ставке`,
    };
  }

  return {
    description: `${playerName}: ${action_type}`,
  };
};

export const GameHistoryModal: React.FC<GameHistoryModalProps> = ({ history, onClose }) => {
  if (!history) {
    return null;
  }

  const { game, players, actions } = history;

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[1000]"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-lg p-5 max-w-4xl max-h-[90vh] w-[90%] flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex justify-between items-center mb-5 flex-shrink-0">
          <h2 className="m-0 text-2xl font-bold text-gray-800">История игры #{game.id}</h2>
          <button
            onClick={onClose}
            className="bg-red-600 hover:bg-red-700 text-white border-none rounded px-4 py-2 cursor-pointer text-base transition-colors"
          >
            ✕ Закрыть
          </button>
        </div>

        <div className="mb-5 p-4 bg-gray-100 rounded-lg flex-shrink-0">
          <div className="grid grid-cols-2 gap-2.5">
            <div>
              <strong>Создана:</strong>{' '}
              {game.created_at ? new Date(game.created_at).toLocaleString('ru-RU') : '-'}
            </div>
            <div>
              <strong>Завершена:</strong>{' '}
              {game.finished_at ? new Date(game.finished_at).toLocaleString('ru-RU') : '-'}
            </div>
            <div>
              <strong>Победитель:</strong>{' '}
              {game.winner !== null ? getPlayerName(game.winner) : '-'}
            </div>
            <div>
              <strong>Игроков:</strong> {game.num_players} • <strong>Ходов:</strong> {actions.length}
            </div>
          </div>
          <div className="mt-4 pt-4 border-t border-gray-300">
            <strong>Игроки:</strong>
            <div className="flex flex-wrap gap-2.5 mt-2">
              {players
                .sort((a, b) => a.player_id - b.player_id)
                .map((player) => (
                  <div
                    key={player.player_id}
                    className={`px-3 py-1.5 rounded text-sm ${
                      player.player_id === 0 ? 'bg-blue-100' : 'bg-gray-100'
                    }`}
                  >
                    {getPlayerName(player.player_id)}
                  </div>
                ))}
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto border border-gray-300 rounded-lg p-4">
          <h3 className="mt-0 text-xl font-semibold text-gray-800 mb-4">Ходы игры</h3>
          {actions.length === 0 ? (
            <div className="text-gray-500 italic">Нет ходов</div>
          ) : (
            <div>
              {actions.map((action, index) => {
                const actionInfo = formatActionDescription(action, players);
                const isHuman = action.player_id === 0;
                const backgroundColor = isHuman
                  ? index % 2 === 0
                    ? 'bg-blue-50'
                    : 'bg-blue-100'
                  : index % 2 === 0
                  ? 'bg-gray-50'
                  : 'bg-white';

                return (
                  <div
                    key={action.id}
                    className={`p-3 border-b border-gray-200 ${backgroundColor} mb-1 rounded`}
                  >
                    <div className="font-bold mb-1 text-gray-800">
                      {actionInfo.description}
                    </div>
                    <div className="text-sm text-gray-600">
                      Ход #{action.turn_number} •{' '}
                      {action.timestamp
                        ? new Date(action.timestamp).toLocaleTimeString('ru-RU')
                        : ''}
                    </div>
                    {action.action_data && Object.keys(action.action_data).length > 0 && (
                      <div className="text-sm text-gray-600 mt-1 p-2 bg-black bg-opacity-5 rounded">
                        {action.action_data.quantity !== null &&
                          action.action_data.quantity !== undefined && (
                            <div>
                              <strong>Количество:</strong> {action.action_data.quantity}
                            </div>
                          )}
                        {action.action_data.value !== null &&
                          action.action_data.value !== undefined && (
                            <div>
                              <strong>Значение:</strong> {action.action_data.value}
                            </div>
                          )}
                        {actionInfo.details && <div>{actionInfo.details}</div>}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

