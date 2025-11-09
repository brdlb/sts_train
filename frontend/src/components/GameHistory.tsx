import React from 'react';
import { ExtendedActionHistoryEntry } from '../services/api';

interface GameHistoryProps {
  bidHistory: Array<[number, number, number]>;
  currentBid: [number, number] | null;
  extendedActionHistory?: ExtendedActionHistoryEntry[];
}

const getPlayerName = (playerId: number): string => {
  if (playerId === 0) return 'You (Human)';
  return `AI Player ${playerId}`;
};

const formatActionDescription = (entry: ExtendedActionHistoryEntry): string => {
  const playerName = getPlayerName(entry.player_id);
  const { action_type, action_data, consequences } = entry;

  if (action_type === 'bid') {
    return `${playerName} сделал ставку: ${action_data.quantity}x${action_data.value}`;
  } else if (action_type === 'challenge') {
    const bidInfo = consequences.bid_quantity && consequences.bid_value
      ? ` ставку ${consequences.bid_quantity}x${consequences.bid_value}`
      : ' ставку';
    const bidderName = consequences.bidder_id !== null && consequences.bidder_id !== undefined
      ? getPlayerName(consequences.bidder_id)
      : 'другого игрока';
    
    if (consequences.challenge_success === true) {
      return `${playerName} оспорил${bidInfo} ${bidderName}. Ставка была неправильной! ${bidderName} проиграл кубик.`;
    } else if (consequences.challenge_success === false) {
      return `${playerName} оспорил${bidInfo} ${bidderName}. Ставка была правильной! ${playerName} проиграл кубик.`;
    } else {
      return `${playerName} оспорил${bidInfo} ${bidderName}.`;
    }
  } else if (action_type === 'believe') {
    const bidInfo = consequences.bid_quantity && consequences.bid_value
      ? ` ставку ${consequences.bid_quantity}x${consequences.bid_value}`
      : ' ставку';
    const bidderName = consequences.bidder_id !== null && consequences.bidder_id !== undefined
      ? getPlayerName(consequences.bidder_id)
      : 'другого игрока';
    
    if (consequences.believe_success === true) {
      return `${playerName} поверил${bidInfo} ${bidderName}. Ставка была точной! ${playerName} получил преимущество.`;
    } else if (consequences.believe_success === false) {
      return `${playerName} поверил${bidInfo} ${bidderName}. Ставка была неточной! ${playerName} проиграл кубик.`;
    } else {
      return `${playerName} поверил${bidInfo} ${bidderName}.`;
    }
  }
  
  return `${playerName}: ${action_type}`;
};

export const GameHistory: React.FC<GameHistoryProps> = ({ bidHistory, currentBid, extendedActionHistory }) => {
  const playerNames = ['You (Human)', 'AI Player 1', 'AI Player 2', 'AI Player 3'];

  return (
    <div style={{ padding: '20px', border: '1px solid #ccc', borderRadius: '8px', marginTop: '20px' }}>
      <h3>История действий</h3>
      <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
        {extendedActionHistory && extendedActionHistory.length > 0 ? (
          <div>
            {extendedActionHistory.map((entry, index) => {
              const actionDescription = formatActionDescription(entry);
              const isHuman = entry.player_id === 0;
              const backgroundColor = isHuman 
                ? (index % 2 === 0 ? '#e3f2fd' : '#bbdefb')
                : (index % 2 === 0 ? '#f9f9f9' : '#fff');
              
              // Determine if there were consequences
              const hasConsequences = entry.consequences.dice_lost !== null && entry.consequences.dice_lost > 0;
              const consequenceColor = entry.consequences.challenge_success === true || entry.consequences.believe_success === true
                ? '#4caf50' // Green for success
                : entry.consequences.challenge_success === false || entry.consequences.believe_success === false
                ? '#f44336' // Red for failure
                : '#666'; // Gray for neutral

              return (
                <div
                  key={index}
                  style={{
                    padding: '12px',
                    borderBottom: '1px solid #eee',
                    backgroundColor,
                    marginBottom: '4px',
                    borderRadius: '4px',
                  }}
                >
                  <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>
                    {actionDescription}
                  </div>
                  {hasConsequences && entry.consequences.loser_id !== null && (
                    <div style={{ fontSize: '0.9em', color: consequenceColor, marginTop: '4px' }}>
                      {getPlayerName(entry.consequences.loser_id)} потерял {entry.consequences.dice_lost} кубик(ов)
                    </div>
                  )}
                  {entry.consequences.actual_count !== null && (
                    <div style={{ fontSize: '0.85em', color: '#666', marginTop: '2px' }}>
                      Фактическое количество: {entry.consequences.actual_count}
                    </div>
                  )}
                  {entry.consequences.action_valid === false && (
                    <div style={{ fontSize: '0.85em', color: '#f44336', marginTop: '2px' }}>
                      Недействительное действие: {entry.consequences.error_msg || 'Неизвестная ошибка'}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          // Fallback to simple bid history if extended history is not available
          <>
            {bidHistory.length === 0 && !currentBid && (
              <div style={{ color: '#666', fontStyle: 'italic' }}>Нет действий пока</div>
            )}
            {bidHistory.map((bid, index) => {
              const [playerId, quantity, value] = bid;
              return (
                <div
                  key={index}
                  style={{
                    padding: '8px',
                    borderBottom: '1px solid #eee',
                    backgroundColor: index % 2 === 0 ? '#f9f9f9' : '#fff',
                  }}
                >
                  <strong>{playerNames[playerId] || `Player ${playerId}`}:</strong> {quantity}x{value}
                </div>
              );
            })}
            {currentBid && (
              <div
                style={{
                  padding: '8px',
                  backgroundColor: '#e3f2fd',
                  fontWeight: 'bold',
                }}
              >
                <strong>Текущая ставка:</strong> {currentBid[0]}x{currentBid[1]}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

