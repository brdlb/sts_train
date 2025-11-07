import React from 'react';

interface DiceDisplayProps {
  dice: number[];
  playerId: number;
  playerName: string;
  isCurrentPlayer: boolean;
  isHuman: boolean;
  diceCount: number;
}

export const DiceDisplay: React.FC<DiceDisplayProps> = ({
  dice,
  playerId,
  playerName,
  isCurrentPlayer,
  isHuman,
  diceCount,
}) => {
  return (
    <div
      style={{
        border: isCurrentPlayer ? '3px solid #4CAF50' : '1px solid #ccc',
        borderRadius: '8px',
        padding: '15px',
        margin: '10px',
        backgroundColor: isCurrentPlayer ? '#f0f8f0' : '#fff',
        minWidth: '200px',
      }}
    >
      <div style={{ fontWeight: 'bold', marginBottom: '10px' }}>
        {playerName} {isCurrentPlayer && '(Current Turn)'}
      </div>
      <div style={{ marginBottom: '10px' }}>
        <strong>Dice Count:</strong> {diceCount}
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px' }}>
        {isHuman ? (
          // Show actual dice for human player
          dice.map((value, index) => (
            <div
              key={index}
              style={{
                width: '40px',
                height: '40px',
                border: '2px solid #333',
                borderRadius: '5px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '20px',
                fontWeight: 'bold',
                backgroundColor: '#fff',
              }}
            >
              {value}
            </div>
          ))
        ) : (
          // Show hidden dice for AI players
          Array.from({ length: diceCount }).map((_, index) => (
            <div
              key={index}
              style={{
                width: '40px',
                height: '40px',
                border: '2px solid #999',
                borderRadius: '5px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '20px',
                backgroundColor: '#f0f0f0',
                color: '#999',
              }}
            >
              ?
            </div>
          ))
        )}
      </div>
    </div>
  );
};

