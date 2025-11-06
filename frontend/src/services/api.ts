import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

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

export interface GameState {
  game_id: string;
  current_player: number;
  turn_number: number;
  game_over: boolean;
  winner: number | null;
  player_dice_count: number[];
  current_bid: [number, number] | null;
  bid_history: Array<[number, number, number]>;
  palifico_active: boolean[];
  believe_called: boolean;
  player_dice: {
    bid_history: number[][];
    static_info: number[];
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
    const response = await api.get<ModelInfo[]>('/api/models/list');
    return response.data;
  },

  getInfo: async (modelId: string): Promise<ModelInfo> => {
    const response = await api.get<ModelInfo>(`/api/models/${modelId}/info`);
    return response.data;
  },

  validate: async (modelPath: string): Promise<{ valid: boolean; path: string }> => {
    const response = await api.post('/api/models/validate', null, {
      params: { model_path: modelPath },
    });
    return response.data;
  },
};

export const gamesApi = {
  create: async (request: CreateGameRequest): Promise<{ game_id: string; state: GameState }> => {
    const response = await api.post<{ game_id: string; state: GameState }>(
      '/api/games/create',
      request
    );
    return response.data;
  },

  getState: async (gameId: string): Promise<GameState> => {
    const response = await api.get<GameState>(`/api/games/${gameId}`);
    return response.data;
  },

  makeAction: async (gameId: string, action: number): Promise<ActionResult> => {
    const response = await api.post<ActionResult>(`/api/games/${gameId}/action`, {
      action,
    });
    return response.data;
  },

  getHistory: async (gameId: string): Promise<GameHistory> => {
    const response = await api.get<GameHistory>(`/api/games/${gameId}/history`);
    return response.data;
  },

  list: async (filters?: { finished?: boolean; limit?: number }): Promise<any[]> => {
    const response = await api.get('/api/games', { params: filters });
    return response.data;
  },
};

export const statisticsApi = {
  getGames: async (): Promise<{
    total_games: number;
    finished_games: number;
    active_games: number;
  }> => {
    const response = await api.get('/api/statistics/games');
    return response.data;
  },

  getPlayer: async (): Promise<PlayerStatistics> => {
    const response = await api.get<PlayerStatistics>('/api/statistics/player');
    return response.data;
  },

  getModels: async (): Promise<ModelStatistics> => {
    const response = await api.get<ModelStatistics>('/api/statistics/models');
    return response.data;
  },
};

