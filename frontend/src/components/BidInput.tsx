import React, { useState } from 'react';

interface BidInputProps {
  currentBid: [number, number] | null;
  maxQuantity: number;
  onBid: (quantity: number, value: number) => void;
  onChallenge: () => void;
  onBelieve: () => void;
  canChallenge: boolean;
  canBelieve: boolean;
  disabled: boolean;
}

export const BidInput: React.FC<BidInputProps> = ({
  currentBid,
  maxQuantity,
  onBid,
  onChallenge,
  onBelieve,
  canChallenge,
  canBelieve,
  disabled,
}) => {
  const [quantity, setQuantity] = useState(1);
  const [value, setValue] = useState(1);

  const handleBid = () => {
    if (quantity >= 1 && quantity <= maxQuantity && value >= 1 && value <= 6) {
      // Validate bid is higher than current bid
      if (currentBid) {
        const [currentQty, currentVal] = currentBid;
        if (quantity < currentQty || (quantity === currentQty && value <= currentVal)) {
          alert('Bid must be higher than current bid!');
          return;
        }
      }
      onBid(quantity, value);
    }
  };

  return (
    <div style={{ padding: '20px', border: '1px solid #ccc', borderRadius: '8px', marginTop: '20px' }}>
      <h3>Your Turn - Make a Move</h3>

      {currentBid && (
        <div style={{ marginBottom: '15px', padding: '10px', backgroundColor: '#fff3cd', borderRadius: '5px' }}>
          <strong>Current Bid:</strong> {currentBid[0]}x{currentBid[1]}
        </div>
      )}

      <div style={{ marginBottom: '15px' }}>
        <label style={{ marginRight: '10px' }}>
          Quantity:
          <input
            type="number"
            min="1"
            max={maxQuantity}
            value={quantity}
            onChange={(e) => setQuantity(parseInt(e.target.value) || 1)}
            disabled={disabled}
            style={{ marginLeft: '5px', padding: '5px', width: '80px' }}
          />
        </label>
        <label style={{ marginLeft: '20px' }}>
          Value (1-6):
          <input
            type="number"
            min="1"
            max="6"
            value={value}
            onChange={(e) => setValue(parseInt(e.target.value) || 1)}
            disabled={disabled}
            style={{ marginLeft: '5px', padding: '5px', width: '80px' }}
          />
        </label>
        <button
          onClick={handleBid}
          disabled={disabled}
          style={{
            marginLeft: '20px',
            padding: '8px 16px',
            backgroundColor: disabled ? '#ccc' : '#2196F3',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: disabled ? 'not-allowed' : 'pointer',
          }}
        >
          Make Bid
        </button>
      </div>

      <div style={{ display: 'flex', gap: '10px' }}>
        <button
          onClick={onChallenge}
          disabled={!canChallenge || disabled}
          style={{
            padding: '10px 20px',
            backgroundColor: canChallenge && !disabled ? '#f44336' : '#ccc',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: canChallenge && !disabled ? 'pointer' : 'not-allowed',
            fontSize: '16px',
          }}
        >
          Challenge
        </button>
        <button
          onClick={onBelieve}
          disabled={!canBelieve || disabled}
          style={{
            padding: '10px 20px',
            backgroundColor: canBelieve && !disabled ? '#FF9800' : '#ccc',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: canBelieve && !disabled ? 'pointer' : 'not-allowed',
            fontSize: '16px',
          }}
        >
          Believe
        </button>
      </div>

      {disabled && (
        <div style={{ marginTop: '10px', color: '#666', fontStyle: 'italic' }}>
          Waiting for AI players...
        </div>
      )}
    </div>
  );
};

