import React from 'react';
import { ExtendedActionHistoryEntry } from '../services/api';
import { getPlayerName } from '../utils/playerHelpers';

interface GameHistoryProps {
  bidHistory: Array<[number, number, number]>;
  currentBid: [number, number] | null;
  extendedActionHistory?: ExtendedActionHistoryEntry[];
}

const formatActionDescription = (entry: ExtendedActionHistoryEntry): string => {
  const playerName = getPlayerName(entry.player_id);
  const { action_type, action_data, consequences } = entry;

  // Ensure consequences exists
  if (!consequences) {
    return `${playerName}: ${action_type}`;
  }

  if (action_type === 'bid') {
    const qty = action_data?.quantity ?? '?';
    const val = action_data?.value ?? '?';
    return `${playerName} сделал ставку: ${qty}x${val}`;
  } else if (action_type === 'challenge') {
    const bidInfo = (consequences.bid_quantity && consequences.bid_value)
      ? ` ставку ${consequences.bid_quantity}x${consequences.bid_value}`
      : ' ставку';
    const bidderName = (consequences.bidder_id !== null && consequences.bidder_id !== undefined && typeof consequences.bidder_id === 'number')
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
    const bidInfo = (consequences.bid_quantity && consequences.bid_value)
      ? ` ставку ${consequences.bid_quantity}x${consequences.bid_value}`
      : ' ставку';
    const bidderName = (consequences.bidder_id !== null && consequences.bidder_id !== undefined && typeof consequences.bidder_id === 'number')
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
    <div className="w-full h-full bg-gray-900/80 rounded-lg p-4 shadow-lg flex flex-col">
      <h3 className="text-xl font-semibold text-yellow-300 pb-2 mb-2 flex-shrink-0">Action History</h3>
      <div className="flex-grow overflow-y-auto pr-2">
        {extendedActionHistory && extendedActionHistory.length > 0 ? (
          <div className="space-y-2">
            {extendedActionHistory.filter(entry => entry && entry.consequences).slice().reverse().map((entry, index) => {
              // Validate entry
              if (!entry || !entry.consequences) {
                return null;
              }

              const actionDescription = formatActionDescription(entry);
              const isHuman = entry.player_id === 0;
              const bgColor = isHuman 
                ? (index % 2 === 0 ? 'bg-blue-900/30' : 'bg-blue-800/30')
                : (index % 2 === 0 ? 'bg-gray-800/50' : 'bg-gray-700/50');
              
              // Determine if there were consequences
              const hasConsequences = entry.consequences && entry.consequences.dice_lost !== null && entry.consequences.dice_lost > 0;
              const consequenceColor = (entry.consequences?.challenge_success === true || entry.consequences?.believe_success === true)
                ? 'text-green-400' // Green for success
                : (entry.consequences?.challenge_success === false || entry.consequences?.believe_success === false)
                ? 'text-red-400' // Red for failure
                : 'text-gray-400'; // Gray for neutral

              // Use a unique key based on entry data
              const uniqueKey = `${entry.player_id}-${entry.action_type}-${entry.consequences.bid_quantity || ''}-${entry.consequences.bid_value || ''}-${index}`;

              return (
                <div
                  key={uniqueKey}
                  className={`p-3 rounded-lg ${bgColor} transition-colors`}
                >
                  <div className="font-semibold text-white mb-1">
                    {actionDescription}
                  </div>
                  {hasConsequences && entry.consequences.loser_id !== null && entry.consequences.loser_id !== undefined && typeof entry.consequences.loser_id === 'number' && (
                    <div className={`text-sm ${consequenceColor} mt-1`}>
                      {getPlayerName(entry.consequences.loser_id)} lost {entry.consequences.dice_lost} die/dice
                    </div>
                  )}
                  {entry.consequences.actual_count !== null && entry.consequences.actual_count !== undefined && (
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
            {currentBid && (
              <div className="p-2 rounded bg-blue-900/30 font-bold text-blue-300">
                <strong>Current Bid:</strong> {currentBid[0]}x{currentBid[1]}
              </div>
            )}
            {bidHistory.slice().reverse().map((bid, index) => {
              const [playerId, quantity, value] = bid;
              return (
                <div
                  key={`${playerId}-${quantity}-${value}-${index}`}
                  className={`p-2 rounded ${index % 2 === 0 ? 'bg-gray-800/50' : 'bg-gray-700/50'}`}
                >
                  <strong className="text-white">{playerNames[playerId] || `Player ${playerId}`}:</strong> 
                  <span className="text-gray-300 ml-2">{quantity}x{value}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

