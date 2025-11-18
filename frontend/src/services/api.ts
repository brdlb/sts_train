import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export interface ModelInfo {
  id: string;
  path: string;
  step: number | null;
  elo: number | null;
  winrate: number | null;
  source: string;
}

export interface ActionConsequences {
  action_valid: boolean;
  dice_lost: number | null;
  loser_id: number | null;
  challenge_success: boolean | null;
  believe_success: boolean | null;
  actual_count: number | null;
  bid_quantity: number | null;
  bid_value: number | null;
  bidder_id: number | null;
  error_msg: string | null;
  player_dice_count_after: number[];
  all_player_dice?: number[][]; // All player dice values during reveal (challenge/believe only)
}

export interface ExtendedActionHistoryEntry {
  player_id: number;
  action_type: string;
  action_data: {
    action_type: string;
    quantity: number | null;
    value: number | null;
  };
  consequences: ActionConsequences;
  turn_number: number;
}

export interface GameState {
  game_id: string;
  current_player: number;
  turn_number: number;
  game_over: boolean;
  winner: number | null;
  player_dice_count: number[];
  current_bid: [number, number] | null;
  bid_history: Array<[number, number, number]>;
  extended_action_history?: ExtendedActionHistoryEntry[];
  palifico_active: boolean[];
  believe_called: boolean;
  last_bid_player_id?: number | null;
  player_dice: {
    bid_history: number[][];
    static_info: number[];
    dice_values?: number[];
  };
  public_info: any;
}

export interface CreateGameRequest {
  model_paths: string[];
}

export interface ActionRequest {
  action: number;
}

export interface ActionResult {
  success: boolean;
  action: {
    action_type: string;
    quantity: number | null;
    value: number | null;
  };
  reward: number;
  game_over: boolean;
  winner: number | null;
  ai_actions?: Array<{
    player_id: number;
    action: {
      action_type: string;
      quantity: number | null;
      value: number | null;
    };
    reward: number;
  }>;
  state: GameState;
}

export interface GameHistory {
  game: {
    id: number;
    created_at: string;
    finished_at: string | null;
    winner: number | null;
    num_players: number;
    is_finished: boolean;
  };
  players: Array<{
    player_id: number;
    player_type: string;
    model_path: string | null;
  }>;
  actions: Array<{
    id: number;
    player_id: number;
    action_type: string;
    action_data: any;
    timestamp: string;
    turn_number: number;
  }>;
  states: Array<{
    turn_number: number;
    state_json: any;
    timestamp: string;
  }>;
}

export interface PlayerStatistics {
  total_games: number;
  games_won: number;
  winrate: number;
  avg_duration_seconds: number;
}

export interface ModelStatistics {
  [modelPath: string]: {
    games_count: number;
    wins_count: number;
    winrate: number;
  };
}

// API functions
export const modelsApi = {
  list: async (): Promise<ModelInfo[]> => {
    const response = await api.get<ModelInfo[]>('/models/list');
    return response.data;
  },

  getInfo: async (modelId: string): Promise<ModelInfo> => {
    const response = await api.get<ModelInfo>(`/models/${modelId}/info`);
    return response.data;
  },

  validate: async (modelPath: string): Promise<{ valid: boolean; path: string }> => {
    const response = await api.post('/models/validate', null, {
      params: { model_path: modelPath },
    });
    return response.data;
  },
};

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

  getHistory: async (gameId: string): Promise<GameHistory> => {
    const response = await api.get<GameHistory>(`/games/${gameId}/history`);
    return response.data;
  },

  getHistoryByDbId: async (dbGameId: number): Promise<GameHistory> => {
    const response = await api.get<GameHistory>(`/games/db/${dbGameId}/history`);
    return response.data;
  },

  list: async (filters?: { finished?: boolean; limit?: number }): Promise<any[]> => {
    const response = await api.get('/games', { params: filters });
    return response.data;
  },

  subscribeToAiTurns: (
    gameId: string,
    onTurn: (data: {
      type: string;
      player_id?: number;
      action?: any;
      reward?: number;
      state?: GameState;
      game_over?: boolean;
      winner?: number | null;
      error?: string;
    }) => void,
    onError?: (error: Event) => void
  ): EventSource => {
    const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';
    const eventSource = new EventSource(`${API_BASE_URL}/games/${gameId}/ai-turns`);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onTurn(data);
        
        // Close connection when done
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

export const statisticsApi = {
  getGames: async (): Promise<{
    total_games: number;
    finished_games: number;
    active_games: number;
  }> => {
    const response = await api.get('/statistics/games');
    return response.data;
  },

  getPlayer: async (): Promise<PlayerStatistics> => {
    const response = await api.get<PlayerStatistics>('/statistics/player');
    return response.data;
  },

  getModels: async (): Promise<ModelStatistics> => {
    const response = await api.get<ModelStatistics>('/statistics/models');
    return response.data;
  },
};

