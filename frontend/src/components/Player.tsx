
import React, { forwardRef } from 'react';
import { Player as PlayerType, Bid } from '../services/api';
import { Dice } from './Dice';

interface PlayerProps {
  player: PlayerType;
  isCurrent: boolean;
  gamePhase: 'BIDDING' | 'REVEAL' | 'ROUND_OVER' | 'GAME_OVER';
  lastBid: Bid | null;
  isLastBidder: boolean;
}

export const Player = forwardRef<HTMLDivElement, PlayerProps>(({ player, isCurrent, gamePhase, lastBid, isLastBidder }, ref) => {
  const isRevealing = gamePhase === 'REVEAL' || gamePhase === 'ROUND_OVER';
  const hasLost = player.dice_count === 0;

  return (
    <div ref={ref} className={`relative p-4 rounded-lg transition-all duration-300 ${isCurrent ? 'bg-yellow-500/20 ring-2 ring-yellow-400' : 'bg-gray-700/50'} ${hasLost ? 'opacity-40' : ''}`}>
      <div className="flex items-center space-x-4">
        <img src={player.avatar} alt={player.name} className="w-16 h-16 rounded-full border-2 border-gray-400" />
        <div>
          <h3 className="text-2xl font-bold text-white">{player.name}</h3>
          {hasLost && <p className="text-lg text-red-400 font-semibold">Eliminated</p>}
        </div>
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        {player.dice.map((d, i) => (
          <Dice
            key={i}
            value={d}
            size={10}
            revealed={isRevealing || player.is_human}
          />
        ))}
      </div>
      {isLastBidder && lastBid ? (
         <div className="absolute -top-2 -right-2 bg-blue-600 text-white pl-3 pr-2 py-2 rounded-full text-4xl font-bold shadow-lg flex items-center">
          <span className="mr-0.5">{lastBid.quantity}</span>
          <span className="mr-1">x</span>
          <Dice
            value={lastBid.face}
            size={8}
          />
        </div>
      ) : null}
    </div>
  );
});

Player.displayName = 'Player';
