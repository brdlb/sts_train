
import React, { useState, useEffect, useMemo, useCallback } from 'react';
import type { Bid, DiceValue } from '../types';
import Dice from './Dice';
import clsx from 'clsx';

interface BidControlsProps {
  currentBid: Bid | null;
  onPlaceBid: (bid: Bid) => void;
  onDudo: () => void;
  onCalza: () => void;
  totalDiceInPlay: number;
  isDisabled: boolean;
  isSpecialRound: boolean;
  playerDiceCount: number;
}

const BidControls: React.FC<BidControlsProps> = ({ currentBid, onPlaceBid, onDudo, onCalza, totalDiceInPlay, isDisabled, isSpecialRound, playerDiceCount }) => {
  const [quantity, setQuantity] = useState(1);
  const [face, setFace] = useState<DiceValue | null>(null);
  const [error, setError] = useState<string | null>(null);

  const getMinimalQuantityForFace = useCallback((selectedFace: DiceValue | null): number => {
    if (!selectedFace) return 1;

    if (isSpecialRound) {
        return currentBid ? currentBid.quantity + 1 : 1;
    }
    
    if (!currentBid) {
        if (playerDiceCount > 2) {
            return Math.max(1, Math.floor(totalDiceInPlay / 5));
        }
        return 1;
    }

    const isCurrentBidOnes = currentBid.face === 1;
    const isNewBidOnes = selectedFace === 1;

    if (isCurrentBidOnes && !isNewBidOnes) {
      return currentBid.quantity * 2 + 1;
    }
    
    if (!isCurrentBidOnes && isNewBidOnes) {
      return Math.ceil(currentBid.quantity / 2);
    }
    
    if (selectedFace > currentBid.face) {
      return currentBid.quantity;
    }
    
    return currentBid.quantity + 1;
  }, [currentBid, isSpecialRound, playerDiceCount, totalDiceInPlay]);

  const minimalQuantity = useMemo(() => getMinimalQuantityForFace(face), [face, getMinimalQuantityForFace]);
  
  const isAboveMinimal = face ? quantity > minimalQuantity : false;

  useEffect(() => {
    if (isDisabled) return;
    
    setError(null);

    if (isSpecialRound && currentBid) {
        setQuantity(currentBid.quantity + 1);
        setFace(currentBid.face);
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
            setQuantity(currentBid.quantity); 
        }
    }
  }, [isDisabled, currentBid, isSpecialRound, playerDiceCount, totalDiceInPlay]);

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
      setError('Пожалуйста, выберите значение кости.');
      return;
    }

    const newBid: Bid = { quantity, face };

    if (isSpecialRound) {
        if (!currentBid) {
             if (newBid.quantity !== 1) {
                setError('Первая ставка в Специальном раунде должна быть 1.');
                return;
             }
             onPlaceBid(newBid);
             return;
        }
        if (newBid.face !== currentBid.face) {
            setError('В Специальном раунде нельзя менять номинал ставки.');
            return;
        }
        if (newBid.quantity <= currentBid.quantity) {
            setError('В Специальном раунде можно только повышать количество.');
            return;
        }
        onPlaceBid(newBid);
        return;
    }

    if (!currentBid) {
      if (face === 1) {
        setError('Нельзя начинать раунд со ставки на единицы (джокеры).');
        return;
      }
      onPlaceBid(newBid);
      return;
    }

    const isCurrentBidOnes = currentBid.face === 1;
    const isNewBidOnes = newBid.face === 1;

    if (isCurrentBidOnes && !isNewBidOnes) {
      const requiredQuantity = currentBid.quantity * 2 + 1;
      if (newBid.quantity >= requiredQuantity) {
        onPlaceBid(newBid);
      } else {
        setError(`После ставки на единицы, следующая ставка должна быть не менее ${requiredQuantity} костей.`);
      }
    } else if (!isCurrentBidOnes && isNewBidOnes) {
      const requiredQuantity = Math.ceil(currentBid.quantity / 2);
      if (newBid.quantity >= requiredQuantity) {
        onPlaceBid(newBid);
      } else {
        setError(`Чтобы переключиться на единицы, нужно поставить не менее ${requiredQuantity} костей.`);
      }
    } else {
      if (newBid.quantity > currentBid.quantity) {
        onPlaceBid(newBid);
      } else if (newBid.quantity === currentBid.quantity && newBid.face > currentBid.face) {
        onPlaceBid(newBid);
      } else {
        setError('Ваша ставка должна быть выше текущей.');
      }
    }
  };

  const faces: DiceValue[] = [1, 2, 3, 4, 5, 6];
  const isFaceSelectionDisabled = isDisabled || (isSpecialRound && !!currentBid);
  const isQuantityChangeDisabled = isDisabled || !face || (isSpecialRound && !currentBid);

  return (
    <div className="p-4 md:p-6 lg:p-4 bg-gray-900 rounded-lg shadow-inner w-full max-w-lg mx-auto flex flex-col justify-between lg:h-[20rem]">
      <div> {/* Top group for flex layout */}
        <h3 className="text-xl md:text-2xl font-semibold text-center text-white mb-4">
          {isSpecialRound 
            ? `Ваш ход (Special раунд!), костей в игре: ${totalDiceInPlay}`
            : `Ваш ход, костей в игре: ${totalDiceInPlay}`
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
              onClick={() => setQuantity(q => Math.min(totalDiceInPlay, q + 1))}
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
          disabled={isDisabled || !face}
          className={`w-full text-white font-bold py-3 px-4 rounded-lg transition-colors text-lg ${
            isDisabled || !face
              ? 'bg-slate-500 cursor-not-allowed'
              : 'bg-green-600 hover:bg-green-700'
          }`}
        >
          Сделать ставку
        </button>
        <div className="flex gap-2 md:gap-4">
            <button
              onClick={onCalza}
              disabled={isDisabled || !currentBid}
              className="w-1/3 bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-2 md:px-4 rounded-lg transition-colors disabled:bg-gray-500 disabled:cursor-not-allowed text-base md:text-lg"
            >
              Верю!
            </button>
            <button
              onClick={onDudo}
              disabled={isDisabled || !currentBid}
              className="w-2/3 bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-4 rounded-lg transition-colors disabled:bg-gray-500 disabled:cursor-not-allowed text-lg"
            >
              Не верю!
            </button>
        </div>
      </div>
    </div>
  );
};

export default BidControls;
