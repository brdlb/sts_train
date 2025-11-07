import React from 'react';

interface GameHistoryProps {
  bidHistory: Array<[number, number, number]>;
  currentBid: [number, number] | null;
}

export const GameHistory: React.FC<GameHistoryProps> = ({ bidHistory, currentBid }) => {
  return (
    <div style={{ padding: '20px', border: '1px solid #ccc', borderRadius: '8px', marginTop: '20px' }}>
      <h3>Bid History</h3>
      <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
        {bidHistory.length === 0 && !currentBid && (
          <div style={{ color: '#666', fontStyle: 'italic' }}>No bids yet</div>
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
              <strong>Player {playerId}:</strong> {quantity}x{value}
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
            <strong>Current Bid:</strong> {currentBid[0]}x{currentBid[1]}
          </div>
        )}
      </div>
    </div>
  );
};

