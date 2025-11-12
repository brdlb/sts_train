
import React, { useState, useEffect } from 'react';
import { Bid, DiceValue } from '../services/api';
import { Dice } from './Dice';
import clsx from 'clsx';

interface BidControlsProps {
  currentBid: Bid | null;
  onPlaceBid: (bid: Bid) => void;
  onDudo: () => void;
  totalDiceInPlay: number;
  isDisabled: boolean;
}

export const BidControls: React.FC<BidControlsProps> = ({ currentBid, onPlaceBid, onDudo, totalDiceInPlay, isDisabled }) => {
  const [quantity, setQuantity] = useState(1);
  const [face, setFace] = useState<DiceValue | null>(null);

  useEffect(() => {
    if (currentBid) {
      setQuantity(currentBid.quantity);
      setFace(currentBid.face);
    } else {
      setQuantity(1);
      setFace(null);
    }
  }, [currentBid]);

  const handlePlaceBid = () => {
    if (face) {
      onPlaceBid({ quantity, face });
    }
  };

  const faces: DiceValue[] = [1, 2, 3, 4, 5, 6];

  return (
    <div className="p-4 md:p-6 lg:p-4 bg-gray-900 rounded-lg shadow-inner w-full max-w-lg mx-auto flex flex-col justify-between lg:h-[20rem]">
      <div>
        <h3 className="text-xl md:text-2xl font-semibold text-center text-white mb-4">
          Your Turn, Total Dice: {totalDiceInPlay}
        </h3>
        <div className="flex items-center justify-center gap-3 md:gap-4 mb-2">
          <div className="flex items-baseline flex-shrink-0">
            <span
              className="font-mono font-bold leading-none text-white text-8xl md:text-9xl"
            >
              {quantity}
            </span>
            <span
              className="font-mono font-light leading-none -ml-2 text-white text-7xl md:text-8xl self-baseline"
            >
              X
            </span>
          </div>

          <button
              onClick={() => setQuantity(q => Math.min(totalDiceInPlay, q + 1))}
              disabled={isDisabled || !face}
              className="w-12 h-12 md:w-14 md:h-14 flex-shrink-0 bg-gray-700 text-white rounded-md disabled:opacity-50 text-3xl"
          >+</button>

          <div className="grid grid-cols-3 gap-2">
              {faces.map(f => (
                <button
                  key={f}
                  onClick={() => setFace(f)}
                  disabled={isDisabled}
                  className={clsx(
                    'w-12 h-12 md:w-14 md:h-14 flex items-center justify-center rounded-md transition-colors',
                    face === f
                        ? 'bg-yellow-500 ring-2 ring-offset-2 ring-offset-gray-900 ring-yellow-400'
                        : 'bg-gray-700',
                    isDisabled
                        ? 'opacity-30 cursor-not-allowed'
                        : 'hover:bg-gray-600'
                  )}
                >
                  <Dice
                    value={f}
                    size={10}
                  />
                </button>
              ))}
          </div>
        </div>
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
          Place Bid
        </button>
        <button
          onClick={onDudo}
          disabled={isDisabled || !currentBid}
          className="w-full bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-4 rounded-lg transition-colors disabled:bg-gray-500 disabled:cursor-not-allowed text-lg"
        >
          Dudo!
        </button>
      </div>
    </div>
  );
};
