import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export type DiceValue = 1 | 2 | 3 | 4 | 5 | 6;

export interface Bid {
  quantity: number;
  face: DiceValue;
}

export interface Player {
  id: string;
  name: string;
  dice_count: number;
  dice: DiceValue[];
  is_human: boolean;
  is_current_player: boolean;
  avatar: string;
}

export interface GameState {
  game_id: string;
  players: Player[];
  current_player_id: string;
  last_bidder_id: string | null;
  current_bid: Bid | null;
  total_dice_count: number;
  game_over: boolean;
  winner_id: string | null;
}

export interface CreateGameRequest {
  model_paths: string[];
}

export interface ActionResult {
  success: boolean;
  state: GameState;
}

// API functions
export const gamesApi = {
  create: async (request: CreateGameRequest): Promise<{ game_id: string; state: GameState }> => {
    const response = await api.post<{ game_id: string; state: GameState }>(
      '/games/create',
      request
    );
    return response.data;
  },

  getState: async (gameId: string): Promise<GameState> => {
    const response = await api.get<GameState>(`/games/${gameId}`);
    return response.data;
  },

  makeAction: async (gameId: string, action: number): Promise<ActionResult> => {
    const response = await api.post<ActionResult>(`/games/${gameId}/action`, {
      action,
    });
    return response.data;
  },

  subscribeToAiTurns: (
    gameId: string,
    onTurn: (data: {
      type: string;
      state?: GameState;
      game_over?: boolean;
      winner_id?: string | null;
      error?: string;
    }) => void,
    onError?: (error: Event) => void
  ): EventSource => {
    const eventSource = new EventSource(`${API_BASE_URL}/games/${gameId}/ai-turns`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onTurn(data);
        if (data.type === 'done' || data.type === 'error') {
          eventSource.close();
        }
      } catch (err) {
        console.error('Error parsing SSE message:', err);
      }
    };

    if (onError) {
      eventSource.onerror = onError;
    } else {
      eventSource.onerror = (error) => {
        console.error('SSE error:', error);
        eventSource.close();
      };
    }

    return eventSource;
  },
};
