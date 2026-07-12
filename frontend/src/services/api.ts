import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
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
  player_ids?: number[]; // Player IDs matching all_player_dice; excludes empty room seats
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
  my_player_id?: number;
  current_player: number;
  turn_number: number;
  game_over: boolean;
  winner: number | null;
  player_dice_count: number[];
  player_ids?: number[]; // Players who started this game; excludes empty room seats
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
  awaiting_reveal_confirmation?: boolean; // Flag indicating if waiting for user to continue after reveal
  player_names?: Record<number, string>;
  state_version?: number;
}

export interface RoomSeat {
  player_id: number;
  seat_type: 'empty' | 'human' | 'bot';
  player_name: string | null;
  is_connected: boolean;
}

export interface Room {
  room_id: string;
  join_code: string;
  status: 'lobby' | 'playing' | 'finished';
  seats: RoomSeat[];
  game_id: string | null;
}

export interface RoomEvent {
  type: 'connected' | 'room_updated' | 'game_started' | 'state_updated' | 'action_rejected' | 'resync_required' | 'game_finished' | 'pong';
  room: Room;
  state?: GameState;
  error?: string;
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
    display_name?: string | null;
    seat_type?: string | null;
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

  continueRound: async (gameId: string): Promise<{ success: boolean; state: GameState }> => {
    const response = await api.post<{ success: boolean; state: GameState }>(
      `/games/${gameId}/continue-round`
    );
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

const getWsUrl = (roomId: string, playerToken: string): string => {
  const encodedToken = encodeURIComponent(playerToken);
  if (API_BASE_URL.startsWith('http://') || API_BASE_URL.startsWith('https://')) {
    const url = new URL(API_BASE_URL);
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    url.pathname = '/ws/rooms/' + roomId;
    url.search = `player_token=${encodedToken}`;
    return url.toString();
  }

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws/rooms/${roomId}?player_token=${encodedToken}`;
};

export const roomsApi = {
  create: async (playerName: string): Promise<{ room: Room; player_id: number; player_token: string }> => {
    const response = await api.post<{ room: Room; player_id: number; player_token: string }>('/rooms', { player_name: playerName });
    return response.data;
  },

  get: async (roomId: string): Promise<{ room: Room }> => {
    const response = await api.get<{ room: Room }>(`/rooms/${roomId}`);
    return response.data;
  },

  join: async (
    roomId: string,
    playerName: string,
    playerToken?: string | null
  ): Promise<{ room: Room; player_id: number; player_token: string }> => {
    const response = await api.post(`/rooms/${roomId}/join`, {
      player_name: playerName,
      player_token: playerToken || undefined,
    });
    return response.data;
  },

  start: async (roomId: string, playerToken: string): Promise<{ room: Room; state: GameState }> => {
    const response = await api.post<{ room: Room; state: GameState }>(`/rooms/${roomId}/start`, null, { headers: { Authorization: `Bearer ${playerToken}` } });
    return response.data;
  },

  connect: (
    roomId: string,
    playerToken: string,
    onEvent: (event: RoomEvent) => void,
    onError?: (event: Event) => void
  ): WebSocket => {
    const socket = new WebSocket(getWsUrl(roomId, playerToken));
    socket.onmessage = (event) => {
      onEvent(JSON.parse(event.data));
    };
    socket.onerror = onError || ((event) => console.error('Room socket error:', event));
    return socket;
  },
};

export const statisticsApi = {
  getGames: async (): Promise<{
    total_games: number;
    average_duration_seconds?: number;
  }> => {
    const response = await api.get('/statistics/games');
    return response.data;
  },

};

