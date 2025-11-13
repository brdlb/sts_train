import React, { useState, useEffect, useMemo, useCallback } from 'react';
import Dice from './Dice';

type DiceValue = 1 | 2 | 3 | 4 | 5 | 6;

// Helper function to combine class names
const clsx = (...classes: (string | boolean | undefined)[]): string => {
  return classes.filter(Boolean).join(' ');
};

interface BidControlsProps {
  currentBid: [number, number] | null; // [quantity, face]
  maxQuantity: number;
  onBid: (quantity: number, value: number) => void;
  onChallenge: () => void;
  onBelieve: () => void;
  canChallenge: boolean;
  canBelieve: boolean;
  disabled: boolean;
  totalDiceInPlay?: number;
  playerDiceCount?: number;
  isSpecialRound?: boolean;
}

const BidControls: React.FC<BidControlsProps> = ({ 
  currentBid, 
  maxQuantity,
  onBid, 
  onChallenge, 
  onBelieve,
  canChallenge,
  canBelieve,
  disabled,
  totalDiceInPlay = 20,
  playerDiceCount = 5,
  isSpecialRound = false,
}) => {
  const [quantity, setQuantity] = useState(1);
  const [face, setFace] = useState<DiceValue | null>(null);
  const [error, setError] = useState<string | null>(null);

  const getMinimalQuantityForFace = useCallback((selectedFace: DiceValue | null): number => {
    if (!selectedFace) return 1;

    if (isSpecialRound) {
        return currentBid ? currentBid[0] + 1 : 1;
    }
    
    if (!currentBid) {
        if (playerDiceCount > 2) {
            return Math.max(1, Math.floor(totalDiceInPlay / 5));
        }
        return 1;
    }

    const isCurrentBidOnes = currentBid[1] === 1;
    const isNewBidOnes = selectedFace === 1;

    if (isCurrentBidOnes && !isNewBidOnes) {
      return currentBid[0] * 2 + 1;
    }
    
    if (!isCurrentBidOnes && isNewBidOnes) {
      return Math.ceil(currentBid[0] / 2);
    }
    
    if (selectedFace > currentBid[1]) {
      return currentBid[0];
    }
    
    return currentBid[0] + 1;
  }, [currentBid, isSpecialRound, playerDiceCount, totalDiceInPlay]);

  const minimalQuantity = useMemo(() => getMinimalQuantityForFace(face), [face, getMinimalQuantityForFace]);
  
  const isAboveMinimal = face ? quantity > minimalQuantity : false;

  useEffect(() => {
    if (disabled) return;
    
    setError(null);

    if (isSpecialRound && currentBid) {
        setQuantity(currentBid[0] + 1);
        setFace(currentBid[1] as DiceValue);
    } else {
        setFace(null); 
        if (!currentBid) {
            if (playerDiceCount > 2) {
              const defaultQuantity = Math.max(1, Math.floor(totalDiceInPlay / 5));
              setQuantity(defaultQuantity);
            } else {
              setQuantity(1);
            }
        } else {
            setQuantity(currentBid[0]); 
        }
    }
  }, [disabled, currentBid, isSpecialRound, playerDiceCount, totalDiceInPlay]);

  const handleFaceChange = (newFace: DiceValue) => {
    setFace(newFace);
    setQuantity(getMinimalQuantityForFace(newFace));
    setError(null);
  };
  
  const handleQuantityReset = () => {
    if (face && isAboveMinimal) {
      setQuantity(minimalQuantity);
    }
  };

  const handlePlaceBid = () => {
    setError(null);
    
    if (!face) {
      setError('Please select a dice value.');
      return;
    }

    if (quantity < 1 || quantity > maxQuantity || face < 1 || face > 6) {
      setError('Invalid bid values.');
      return;
    }

    // Validate bid is higher than current bid
    if (currentBid) {
      const [currentQty, currentVal] = currentBid;
      if (quantity < currentQty || (quantity === currentQty && face <= currentVal)) {
        setError('Bid must be higher than current bid!');
        return;
      }
    }

    if (isSpecialRound) {
        if (!currentBid) {
             if (quantity !== 1) {
                setError('First bid in Special round must be 1.');
                return;
             }
             onBid(quantity, face);
             return;
        }
        if (face !== currentBid[1]) {
            setError('Cannot change face value in Special round.');
            return;
        }
        if (quantity <= currentBid[0]) {
            setError('Must increase quantity in Special round.');
            return;
        }
        onBid(quantity, face);
        return;
    }

    if (!currentBid) {
      if (face === 1) {
        setError('Cannot start round with ones (wildcards).');
        return;
      }
      onBid(quantity, face);
      return;
    }

    const isCurrentBidOnes = currentBid[1] === 1;
    const isNewBidOnes = face === 1;

    if (isCurrentBidOnes && !isNewBidOnes) {
      const requiredQuantity = currentBid[0] * 2 + 1;
      if (quantity >= requiredQuantity) {
        onBid(quantity, face);
      } else {
        setError(`After bidding on ones, next bid must be at least ${requiredQuantity} dice.`);
      }
    } else if (!isCurrentBidOnes && isNewBidOnes) {
      const requiredQuantity = Math.ceil(currentBid[0] / 2);
      if (quantity >= requiredQuantity) {
        onBid(quantity, face);
      } else {
        setError(`To switch to ones, must bid at least ${requiredQuantity} dice.`);
      }
    } else {
      if (quantity > currentBid[0]) {
        onBid(quantity, face);
      } else if (quantity === currentBid[0] && face > currentBid[1]) {
        onBid(quantity, face);
      } else {
        setError('Your bid must be higher than the current bid.');
      }
    }
  };

  const faces: DiceValue[] = [1, 2, 3, 4, 5, 6];
  const isFaceSelectionDisabled = disabled || (isSpecialRound && !!currentBid);
  const isQuantityChangeDisabled = disabled || !face || (isSpecialRound && !currentBid);

  return (
    <div className="p-4 md:p-6 lg:p-4 bg-gray-900 rounded-lg shadow-inner w-full max-w-lg mx-auto flex flex-col justify-between lg:h-[20rem]">
      <div>
        <h3 className="text-xl md:text-2xl font-semibold text-center text-white mb-4">
          {isSpecialRound 
            ? `Your Turn (Special Round!), Dice in Play: ${totalDiceInPlay}`
            : `Your Turn, Dice in Play: ${totalDiceInPlay}`
          }
        </h3>
        <div className="flex items-center justify-center gap-3 md:gap-4 mb-2">
          <div className="flex items-baseline flex-shrink-0">
            <span
              onClick={handleQuantityReset}
              className={`font-mono font-bold leading-none cursor-pointer transition-colors ${
                isAboveMinimal ? 'text-red-500' : 'text-white'
              } text-8xl md:text-9xl`}
            >
              {quantity}
            </span>
            <span
              onClick={handleQuantityReset}
              className={`font-mono font-light leading-none -ml-2 cursor-pointer transition-colors ${
                isAboveMinimal ? 'text-red-500' : 'text-white'
              } text-7xl md:text-8xl self-baseline`}
            >
              X
            </span>
          </div>
          
          <button
              onClick={() => setQuantity(q => Math.min(maxQuantity, q + 1))}
              disabled={isQuantityChangeDisabled}
              className="w-12 h-12 md:w-14 md:h-14 flex-shrink-0 bg-gray-700 text-white rounded-md disabled:opacity-50 text-3xl"
          >+</button>
          
          <div className="grid grid-cols-3 gap-2">
              {faces.map(f => (
                <button
                  key={f}
                  onClick={() => handleFaceChange(f)}
                  disabled={isFaceSelectionDisabled}
                  className={clsx(
                    'w-12 h-12 md:w-14 md:h-14 flex items-center justify-center rounded-md transition-colors',
                    face === f 
                        ? 'bg-yellow-500 ring-2 ring-offset-2 ring-offset-gray-900 ring-yellow-400' 
                        : 'bg-gray-700',
                    isFaceSelectionDisabled 
                        ? 'opacity-30 cursor-not-allowed' 
                        : 'hover:bg-gray-600'
                  )}
                >
                  <Dice 
                    value={f} 
                    size={10} 
                    color="bg-transparent" 
                    dotColor={face === f ? 'bg-gray-800' : 'bg-white'}
                    starColor={face === f ? 'text-gray-800' : 'text-white'}
                    noBorder
                  />
                </button>
              ))}
          </div>
        </div>
        
        {error && <p className="text-red-400 text-sm md:text-base text-center h-5">{error}</p>}
      </div>
      
      <div className="space-y-3">
        <button
          onClick={handlePlaceBid}
          disabled={disabled || !face}
          className={`w-full text-white font-bold py-3 px-4 rounded-lg transition-colors text-lg ${
            disabled || !face
              ? 'bg-slate-500 cursor-not-allowed'
              : 'bg-green-600 hover:bg-green-700'
          }`}
        >
          Make Bid
        </button>
        <div className="flex gap-2 md:gap-4">
            <button
              onClick={onBelieve}
              disabled={disabled || !canBelieve}
              className="w-1/3 bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-2 md:px-4 rounded-lg transition-colors disabled:bg-gray-500 disabled:cursor-not-allowed text-base md:text-lg"
            >
              Believe!
            </button>
            <button
              onClick={onChallenge}
              disabled={disabled || !canChallenge}
              className="w-2/3 bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-4 rounded-lg transition-colors disabled:bg-gray-500 disabled:cursor-not-allowed text-lg"
            >
              Challenge!
            </button>
        </div>
      </div>
    </div>
  );
};

export default BidControls;

