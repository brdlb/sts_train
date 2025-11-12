
import React, { forwardRef } from 'react';
import type { Player as PlayerType, Bid } from '../types';
import Dice from './Dice';

interface PlayerProps {
  player: PlayerType;
  isCurrent: boolean;
  gamePhase: 'bidding' | 'reveal' | 'round_over' | 'game_over' | 'special_round_decision';
  lastBid: Bid | null;
  isLastBidder: boolean;
  roundBidHistory: { bid: Bid; bidderId: string; }[];
  isSpecialRound: boolean;
}

const Player = forwardRef<HTMLDivElement, PlayerProps>(({ player, isCurrent, gamePhase, lastBid, isLastBidder, roundBidHistory, isSpecialRound }, ref) => {
  const isRevealing = gamePhase === 'reveal' || gamePhase === 'round_over';
  const hasLost = player.dice.length === 0;
  
  const playerLastBidInHistory = [...roundBidHistory].reverse().find(h => h.bidderId === player.id);


  return (
    <div ref={ref} className={`relative p-4 rounded-lg transition-all duration-300 ${isCurrent ? 'bg-yellow-500/20 ring-2 ring-yellow-400' : 'bg-gray-700/50'} ${hasLost ? 'opacity-40' : ''}`}>
      <div className="flex items-center space-x-4">
        <img src={player.avatar} alt={player.name} className="w-16 h-16 rounded-full border-2 border-gray-400" />
        <div>
          <h3 className="text-2xl font-bold text-white">{player.name} {player.isHuman && '(Вы)'}</h3>
          {hasLost && <p className="text-lg text-red-400 font-semibold">Выбыл</p>}
        </div>
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        {player.dice.map((d, i) => (
          <Dice 
            key={i} 
            value={d} 
            size={10} 
            revealed={isRevealing || player.isHuman} 
            color={player.color}
            bidForHighlighting={lastBid}
            isSpecialRound={isSpecialRound}
            gamePhase={gamePhase}
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
            color="bg-transparent"
            dotColor="bg-white"
            starColor="text-white"
            borderColor="border-white"
            dotSizeClass="w-1.5 h-1.5"
            starSizeClass="text-2xl"
          />
        </div>
      ) : playerLastBidInHistory ? (
         <div className="absolute -top-2 -right-2 bg-gray-500 text-white pl-3 pr-2 py-2 rounded-full text-4xl font-bold shadow-lg opacity-60 flex items-center">
          <span className="mr-0.5">{playerLastBidInHistory.bid.quantity}</span>
          <span className="mr-1">x</span>
           <Dice
            value={playerLastBidInHistory.bid.face}
            size={8}
            color="bg-transparent"
            dotColor="bg-white"
            starColor="text-white"
            borderColor="border-white"
            dotSizeClass="w-1.5 h-1.5"
            starSizeClass="text-2xl"
          />
        </div>
      ) : null}
    </div>
  );
});

Player.displayName = 'Player';

export default Player;