import React, { forwardRef } from 'react';
import Dice from './Dice';
import { PLAYER_COLORS, PLAYER_NAMES } from '../constants';

interface PlayerProps {
  playerId: number;
  playerName: string;
  dice: number[]; // Actual dice values (empty for non-human players unless revealed)
  diceCount: number; // Number of dice
  isCurrent: boolean;
  isHuman: boolean;
  gamePhase?: 'bidding' | 'reveal' | 'round_over' | 'game_over';
  lastBid?: [number, number] | null; // [quantity, face]
  isLastBidder?: boolean;
  bidHistory?: Array<[number, number, number]>; // [player, quantity, value]
  isSpecialRound?: boolean;
  revealed?: boolean; // Whether dice should be revealed (for all players during reveal phase)
}

const Player = forwardRef<HTMLDivElement, PlayerProps>(({ 
  playerId, 
  playerName, 
  dice, 
  diceCount, 
  isCurrent, 
  isHuman,
  gamePhase = 'bidding',
  lastBid,
  isLastBidder = false,
  bidHistory = [],
  isSpecialRound = false,
  revealed = false,
}, ref) => {
  const isRevealing = gamePhase === 'reveal' || gamePhase === 'round_over';
  const hasLost = diceCount === 0;
  
  // Find this player's last bid in history
  const playerLastBidInHistory = [...bidHistory].reverse().find(h => h[0] === playerId);
  
  // Get player color
  const playerColor = PLAYER_COLORS[playerId % PLAYER_COLORS.length];
  
  // Determine if dice should be shown
  const shouldShowDice = isRevealing || isHuman || revealed;
  
  // Create dice array - use actual dice if available, otherwise create placeholder array
  const diceToShow = shouldShowDice && dice.length > 0 
    ? dice 
    : Array(diceCount).fill(0); // Placeholder for hidden dice

  return (
    <div 
      ref={ref} 
      className={`relative p-4 rounded-lg transition-all duration-300 ${
        isCurrent ? 'bg-yellow-500/20 ring-2 ring-yellow-400' : 'bg-gray-700/50'
      } ${hasLost ? 'opacity-40' : ''}`}
    >
      <div className="flex items-center space-x-4">
        <div>
          <h3 className="text-2xl font-bold text-white">
            {playerName} {isHuman && '(You)'}
          </h3>
          {hasLost && <p className="text-lg text-red-400 font-semibold">Eliminated</p>}
        </div>
      </div>
      <div className="mt-4 flex flex-wrap gap-2">
        {diceToShow.map((d, i) => (
          <Dice 
            key={i} 
            value={d as 1 | 2 | 3 | 4 | 5 | 6} 
            size={10} 
            revealed={shouldShowDice && d > 0} 
            color={playerColor}
            bidForHighlighting={lastBid}
            isSpecialRound={isSpecialRound}
            gamePhase={gamePhase}
          />
        ))}
      </div>
      {isLastBidder && lastBid ? (
         <div className="absolute -top-2 -right-2 bg-blue-600 text-white pl-3 pr-2 py-2 rounded-full text-4xl font-bold shadow-lg flex items-center">
          <span className="mr-0.5">{lastBid[0]}</span>
          <span className="mr-1">x</span>
          <Dice
            value={lastBid[1] as 1 | 2 | 3 | 4 | 5 | 6}
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
          <span className="mr-0.5">{playerLastBidInHistory[1]}</span>
          <span className="mr-1">x</span>
           <Dice
            value={playerLastBidInHistory[2] as 1 | 2 | 3 | 4 | 5 | 6}
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

