import React from 'react';
import { GameHistory } from '../services/api';

interface GameHistoryModalProps {
  history: GameHistory | null;
  onClose: () => void;
}

const getPlayerName = (playerId: number, players: GameHistory['players']): string => {
  if (playerId === 0) return 'You (Human)';
  const player = players.find(p => p.player_id === playerId);
  if (player?.player_type === 'ai') {
    const modelName = player.model_path 
      ? player.model_path.split('/').pop()?.replace('.zip', '') || `AI Player ${playerId}`
      : `AI Player ${playerId}`;
    return modelName;
  }
  return `Player ${playerId}`;
};

const formatActionDescription = (
  action: GameHistory['actions'][0],
  players: GameHistory['players']
): { description: string; details?: string } => {
  const playerName = getPlayerName(action.player_id, players);
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
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
      onClick={onClose}
    >
      <div
        style={{
          backgroundColor: 'white',
          borderRadius: '8px',
          padding: '20px',
          maxWidth: '800px',
          maxHeight: '90vh',
          width: '90%',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '20px',
            flexShrink: 0,
          }}
        >
          <h2 style={{ margin: 0 }}>История игры #{game.id}</h2>
          <button
            onClick={onClose}
            style={{
              backgroundColor: '#f44336',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              padding: '8px 16px',
              cursor: 'pointer',
              fontSize: '16px',
            }}
          >
            ✕ Закрыть
          </button>
        </div>

        <div
          style={{
            marginBottom: '20px',
            padding: '15px',
            backgroundColor: '#f5f5f5',
            borderRadius: '5px',
            flexShrink: 0,
          }}
        >
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px' }}>
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
              {game.winner !== null ? getPlayerName(game.winner, players) : '-'}
            </div>
            <div>
              <strong>Игроков:</strong> {game.num_players} • <strong>Ходов:</strong> {actions.length}
            </div>
          </div>
          <div style={{ marginTop: '15px', paddingTop: '15px', borderTop: '1px solid #ddd' }}>
            <strong>Игроки:</strong>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginTop: '8px' }}>
              {players
                .sort((a, b) => a.player_id - b.player_id)
                .map((player) => (
                  <div
                    key={player.player_id}
                    style={{
                      padding: '6px 12px',
                      backgroundColor: player.player_id === 0 ? '#e3f2fd' : '#f5f5f5',
                      borderRadius: '4px',
                      fontSize: '0.9em',
                    }}
                  >
                    {getPlayerName(player.player_id, players)}
                  </div>
                ))}
            </div>
          </div>
        </div>

        <div
          style={{
            flex: 1,
            overflowY: 'auto',
            border: '1px solid #ccc',
            borderRadius: '5px',
            padding: '15px',
          }}
        >
          <h3 style={{ marginTop: 0 }}>Ходы игры</h3>
          {actions.length === 0 ? (
            <div style={{ color: '#666', fontStyle: 'italic' }}>Нет ходов</div>
          ) : (
            <div>
              {actions.map((action, index) => {
                const actionInfo = formatActionDescription(action, players);
                const isHuman = action.player_id === 0;
                const backgroundColor = isHuman
                  ? index % 2 === 0
                    ? '#e3f2fd'
                    : '#bbdefb'
                  : index % 2 === 0
                  ? '#f9f9f9'
                  : '#fff';

                return (
                  <div
                    key={action.id}
                    style={{
                      padding: '12px',
                      borderBottom: '1px solid #eee',
                      backgroundColor,
                      marginBottom: '4px',
                      borderRadius: '4px',
                    }}
                  >
                    <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>
                      {actionInfo.description}
                    </div>
                    <div style={{ fontSize: '0.85em', color: '#666' }}>
                      Ход #{action.turn_number} •{' '}
                      {action.timestamp
                        ? new Date(action.timestamp).toLocaleTimeString('ru-RU')
                        : ''}
                    </div>
                    {action.action_data && Object.keys(action.action_data).length > 0 && (
                      <div
                        style={{
                          fontSize: '0.85em',
                          color: '#666',
                          marginTop: '4px',
                          padding: '8px',
                          backgroundColor: 'rgba(0, 0, 0, 0.05)',
                          borderRadius: '4px',
                        }}
                      >
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

