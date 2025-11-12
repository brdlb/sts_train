export type DiceValue = 1 | 2 | 3 | 4 | 5 | 6;

export type GameStage = 'CHAOS' | 'POSITIVE' | 'TENSE' | 'KNIFE_FIGHT' | 'DUEL';

export interface Bid {
  quantity: number;
  face: DiceValue;
}

export interface Player {
  id: string;
  name: string;
  dice: DiceValue[];
  isHuman: boolean;
  personality?: BotPersonality;
  avatar: string;
  color: string;
  hasUsedSpecialRound: boolean;
}

export interface BotAffinities {
  // Эти числа действуют как множители для результата анализа.
  // 1.0 - стандарт. >1.0 - специальность. <1.0 - слабость.
  firstBidAnalysis: number;
  preRevealAnalysis: number;
  facePatternAnalysis: number;
}

export interface BotPersonality {
  name: string;
  description: string;
  avatar: string;
  skillLevel: 'EASY' | 'MEDIUM' | 'HARD';
  affinities: BotAffinities;
}

export interface LogEntry {
  message: string;
  playerId?: string;
}

export interface PlayerAnalysis {
  id: string;
  // Метрика 1: Анализ первой ставки
  firstBidBluffs: { count: number; total: number };
  // Метрика 2: Анализ ставок перед вскрытием
  preRevealTendency: { bluffCount: number; strongHandCount: number; total: number };
  // Метрика 3: Паттерны блефа по номиналам
  faceBluffPatterns: Record<string, { bluffCount: number; totalBids: number }>; // ключ - DiceValue
}

export interface GameState {
  players: Player[];
  currentPlayerIndex: number;
  currentBid: Bid | null;
  gamePhase: 'bidding' | 'reveal' | 'round_over' | 'game_over' | 'special_round_decision';
  gameLog: LogEntry[];
  roundStarterIndex: number;
  lastBidderIndex: number | null;
  challengeType: 'dudo' | 'calza' | null;
  roundBidHistory: { bid: Bid; bidderId: string; }[];
  roundOutcome: {
    playerToUpdate: { id: string; diceChange: number } | null;
    actualDiceCount: number;
  } | null;
  isSpecialRound: boolean;
  roundNumber: number;
  playerAnalysis: Record<string, PlayerAnalysis>;
}

export interface BotDecision {
  decision: 'BID' | 'DUDO' | 'CALZA';
  bid: Bid | null;
  thought: string;
  dialogue: string;
}

// Google Analytics
export interface GameSessionStats {
  gamesPlayed: number;
  sessionsStarted: number;
}

declare global {
  interface Window {
    gtag?: (command: string, targetId: string, config?: Record<string, any>) => void;
  }
}