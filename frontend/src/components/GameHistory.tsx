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
    <div className="w-full h-full bg-gray-900/80 rounded-lg p-4 shadow-lg flex flex-col border border-gray-700">
      <h3 className="text-xl font-semibold text-yellow-300 border-b border-yellow-300/30 pb-2 mb-2 flex-shrink-0">Action History</h3>
      <div className="flex-grow overflow-y-auto pr-2">
        {extendedActionHistory && extendedActionHistory.length > 0 ? (
          <div className="space-y-2">
            {extendedActionHistory.map((entry, index) => {
              const actionDescription = formatActionDescription(entry);
              const isHuman = entry.player_id === 0;
              const bgColor = isHuman 
                ? (index % 2 === 0 ? 'bg-blue-900/30' : 'bg-blue-800/30')
                : (index % 2 === 0 ? 'bg-gray-800/50' : 'bg-gray-700/50');
              
              // Determine if there were consequences
              const hasConsequences = entry.consequences.dice_lost !== null && entry.consequences.dice_lost > 0;
              const consequenceColor = entry.consequences.challenge_success === true || entry.consequences.believe_success === true
                ? 'text-green-400' // Green for success
                : entry.consequences.challenge_success === false || entry.consequences.believe_success === false
                ? 'text-red-400' // Red for failure
                : 'text-gray-400'; // Gray for neutral

              return (
                <div
                  key={index}
                  className={`p-3 rounded-lg ${bgColor} border border-gray-700/50 transition-colors`}
                >
                  <div className="font-semibold text-white mb-1">
                    {actionDescription}
                  </div>
                  {hasConsequences && entry.consequences.loser_id !== null && (
                    <div className={`text-sm ${consequenceColor} mt-1`}>
                      {getPlayerName(entry.consequences.loser_id)} lost {entry.consequences.dice_lost} die/dice
                    </div>
                  )}
                  {entry.consequences.actual_count !== null && (
                    <div className="text-xs text-gray-400 mt-1">
                      Actual count: {entry.consequences.actual_count}
                    </div>
                  )}
                  {entry.consequences.action_valid === false && (
                    <div className="text-xs text-red-400 mt-1">
                      Invalid action: {entry.consequences.error_msg || 'Unknown error'}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          // Fallback to simple bid history if extended history is not available
          <div className="space-y-2">
            {bidHistory.length === 0 && !currentBid && (
              <div className="text-gray-400 italic">No actions yet</div>
            )}
            {bidHistory.map((bid, index) => {
              const [playerId, quantity, value] = bid;
              return (
                <div
                  key={index}
                  className={`p-2 rounded ${index % 2 === 0 ? 'bg-gray-800/50' : 'bg-gray-700/50'} border border-gray-700/50`}
                >
                  <strong className="text-white">{playerNames[playerId] || `Player ${playerId}`}:</strong> 
                  <span className="text-gray-300 ml-2">{quantity}x{value}</span>
                </div>
              );
            })}
            {currentBid && (
              <div className="p-2 rounded bg-blue-900/30 border border-blue-700/50 font-bold text-blue-300">
                <strong>Current Bid:</strong> {currentBid[0]}x{currentBid[1]}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

