
import React from 'react';
import type { Bid, DiceValue } from '../types';
import clsx from 'clsx';

interface DiceProps {
  value: DiceValue;
  size?: number;
  revealed?: boolean;
  color?: string;
  bidForHighlighting?: Bid | null;
  isSpecialRound?: boolean;
  gamePhase?: string;
  dotColor?: string;
  starColor?: string;
  noBorder?: boolean;
  borderColor?: string;
  dotSizeClass?: string;
  starSizeClass?: string;
}

const Dice: React.FC<DiceProps> = ({ 
  value, 
  size = 10, 
  revealed = true, 
  color = 'bg-red-600', 
  bidForHighlighting, 
  isSpecialRound, 
  gamePhase,
  dotColor = 'bg-white',
  starColor = 'text-white',
  noBorder = false,
  borderColor = 'border-gray-400',
  dotSizeClass = 'w-2 h-2',
  starSizeClass = 'text-3xl',
}) => {
  const baseClasses = `w-${size} h-${size} rounded-lg shadow-lg flex items-center justify-center transition-transform duration-500`;
  const borderClasses = noBorder ? '' : `border-2 ${borderColor}`;

  if (!revealed) {
    return (
      <div className={clsx(baseClasses, borderClasses, 'bg-gray-600 transform hover:scale-105 cursor-pointer')}>
        <svg xmlns="http://www.w3.org/2000/svg" className="h-2/3 w-2/3 text-gray-400" fill="none" viewBox="0 0 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </div>
    );
  }

  const isHighlightPhase = gamePhase === 'reveal' || gamePhase === 'round_over';
  let shouldHighlight = false;

  if (isHighlightPhase && bidForHighlighting) {
    const bidFace = bidForHighlighting.face;
    if (isSpecialRound || bidFace === 1) {
      if (value === bidFace) {
        shouldHighlight = true;
      }
    } else {
      if (value === bidFace || value === 1) {
        shouldHighlight = true;
      }
    }
  }

  const dot = (className?: string) => <div className={clsx(dotSizeClass, "rounded-full", dotColor, className)}></div>;

  const patterns: { [key in DiceValue]: React.ReactNode } = {
    1: <span className={clsx("font-bold leading-none select-none relative bottom-0.5", starColor, starSizeClass)}>★</span>,
    2: (
        <div className="w-full h-full flex justify-between p-1.5">
            {dot('self-start')}
            {dot('self-end')}
        </div>
    ),
    3: (
        <div className="w-full h-full flex justify-between p-1.5">
            {dot('self-start')}
            {dot('self-center')}
            {dot('self-end')}
        </div>
    ),
    4: (
        <div className="w-full h-full flex justify-between p-1.5">
            <div className="flex flex-col justify-between">
                {dot()}
                {dot()}
            </div>
            <div className="flex flex-col justify-between">
                {dot()}
                {dot()}
            </div>
        </div>
    ),
    5: (
        <div className="w-full h-full relative p-1.5">
            <div className="w-full h-full flex justify-between">
                <div className="flex flex-col justify-between">
                    {dot()}
                    {dot()}
                </div>
                <div className="flex flex-col justify-between">
                    {dot()}
                    {dot()}
                </div>
            </div>
            <div className="absolute inset-0 flex justify-center items-center">
                {dot()}
            </div>
        </div>
    ),
    6: (
        <div className="w-full h-full flex justify-between p-1.5">
            <div className="flex flex-col justify-between">
                {dot()}
                {dot()}
                {dot()}
            </div>
            <div className="flex flex-col justify-between">
                {dot()}
                {dot()}
                {dot()}
            </div>
        </div>
    ),
  };
  
  return (
    <div className={clsx(
        baseClasses,
        borderClasses, 
        color, 
        shouldHighlight && 'animate-pulse-highlight brightness-150'
      )}
    >
      {patterns[value]}
    </div>
  );
};

export default Dice;