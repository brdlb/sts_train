CREATE TABLE IF NOT EXISTS games (
  id TEXT PRIMARY KEY,
  room_id TEXT NOT NULL,
  started_at INTEGER NOT NULL,
  finished_at INTEGER,
  winner_seat INTEGER,
  turns INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,
  expires_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS game_players (
  game_id TEXT NOT NULL,
  seat INTEGER NOT NULL,
  display_name TEXT NOT NULL,
  token_hash TEXT NOT NULL,
  is_winner INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (game_id, seat)
);
CREATE TABLE IF NOT EXISTS game_actions (
  game_id TEXT NOT NULL,
  turn_number INTEGER NOT NULL,
  player_seat INTEGER NOT NULL,
  type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  consequences_json TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  PRIMARY KEY (game_id, turn_number)
);
CREATE INDEX IF NOT EXISTS idx_games_expires_at ON games(expires_at);
CREATE INDEX IF NOT EXISTS idx_games_finished_at ON games(finished_at);
CREATE INDEX IF NOT EXISTS idx_game_actions_game_id ON game_actions(game_id);
CREATE INDEX IF NOT EXISTS idx_game_players_game_id ON game_players(game_id);
