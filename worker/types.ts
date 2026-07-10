export type Action = { action_type: 'bid'; quantity: number; value: number } | { action_type: 'challenge' | 'believe' };
export type Player = { seat: number; name: string; tokenHash: string; dice: number[]; diceCount: number; connected: boolean };
export type HistoryEntry = { player_id: number; action_type: Action['action_type']; action_data: { action_type: string; quantity: number | null; value: number | null }; consequences: Record<string, unknown>; turn_number: number };
export type RoomState = {
  roomId: string; joinCode: string; status: 'lobby' | 'playing' | 'finished'; hostSeat: number; players: Array<Player | null>;
  gameId: string | null; currentPlayer: number; currentBid: [number, number] | null; lastBidPlayer: number | null;
  bidHistory: Array<[number, number, number]>; history: HistoryEntry[]; palifico: boolean[]; specialRound: boolean;
  round: number; stateVersion: number; winner: number | null; awaitingReveal: boolean; lastActivity: number; processed: Record<string, unknown>;
};
export interface Env { ROOMS: DurableObjectNamespace; DB: D1Database; ASSETS: Fetcher; VERSION?: string }
