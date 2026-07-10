import { applyAction, continueRound, startGame, viewFor } from './game';
import type { Action, Env, Player, RoomState } from './types';

const json = (body: unknown, status = 200) => Response.json(body, { status });
const hash = async (value: string) => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value)))).map((n) => n.toString(16).padStart(2, '0')).join('');
const publicRoom = (s: RoomState) => ({ room_id: s.roomId, join_code: s.joinCode, status: s.status, game_id: s.gameId, state_version: s.stateVersion, seats: s.players.map((p, player_id) => ({ player_id, seat_type: p ? 'human' : 'empty', player_name: p?.name ?? null, is_connected: p?.connected ?? false })) });
const roomCode = () => crypto.randomUUID().replace(/-/g, '').slice(0, 6).toUpperCase();

export class RoomDurableObject {
  private sockets = new Map<WebSocket, number>();
  constructor(private ctx: DurableObjectState, private env: Env) {}
  private async state(): Promise<RoomState> { const state = await this.ctx.storage.get<RoomState>('state'); if (!state) throw new Error('Room not initialized'); return state; }
  private async save(s: RoomState) { s.lastActivity = Date.now(); await this.ctx.storage.put('state', s); await this.ctx.storage.setAlarm(s.lastActivity + 86_400_000); }
  private token(request: Request) { const auth = request.headers.get('Authorization'); return auth?.startsWith('Bearer ') ? auth.slice(7) : new URL(request.url).searchParams.get('player_token'); }
  private async seat(s: RoomState, raw: string | null) { if (!raw) return -1; const h = await hash(raw); return s.players.findIndex((p) => p?.tokenHash === h); }
  private async broadcast(s: RoomState, type: string) { for (const [socket, seat] of this.sockets) { try { socket.send(JSON.stringify({ type, room: publicRoom(s), state: s.status === 'lobby' ? undefined : viewFor(s, seat) })); } catch { this.sockets.delete(socket); } } }
  async fetch(request: Request): Promise<Response> {
    try {
      const url = new URL(request.url); const parts = url.pathname.split('/').filter(Boolean); let s: RoomState;
      if (request.method === 'POST' && parts.at(-1) === 'initialize') {
        const { room_id, player_name } = await request.json<{ room_id: string; player_name: string }>(); const token = crypto.randomUUID() + crypto.randomUUID();
        s = { roomId: room_id, joinCode: roomCode(), status: 'lobby', hostSeat: 0, players: [{ seat: 0, name: cleanName(player_name), tokenHash: await hash(token), dice: [], diceCount: 5, connected: false }, null, null, null], gameId: null, currentPlayer: 0, currentBid: null, lastBidPlayer: null, bidHistory: [], history: [], palifico: [false, false, false, false], specialRound: false, round: 0, stateVersion: 1, winner: null, awaitingReveal: false, lastActivity: Date.now(), processed: {} };
        await this.save(s); return json({ room: publicRoom(s), player_id: 0, player_token: token }, 201);
      }
      s = await this.state();
      if (request.method === 'GET' && parts.at(-1) === 'room') return json({ room: publicRoom(s) });
      if (request.method === 'POST' && parts.at(-1) === 'join') {
        const body = await request.json<{ player_name: string; player_token?: string }>(); let seat = await this.seat(s, body.player_token ?? null); let token = body.player_token;
        if (seat < 0) { seat = s.players.findIndex((p) => !p); if (seat < 0 || s.status !== 'lobby') return json({ error: 'Room is full or has already started' }, 409); token = crypto.randomUUID() + crypto.randomUUID(); s.players[seat] = { seat, name: cleanName(body.player_name), tokenHash: await hash(token), dice: [], diceCount: 5, connected: false }; s.stateVersion++; await this.save(s); await this.broadcast(s, 'room_updated'); }
        return json({ room: publicRoom(s), player_id: seat, player_token: token });
      }
      const seat = await this.seat(s, this.token(request)); if (seat < 0) return json({ error: 'Unauthorized' }, 401);
      if (request.method === 'POST' && parts.at(-1) === 'start') { if (seat !== s.hostSeat) return json({ error: 'Only the host can start' }, 403); startGame(s); s.stateVersion++; await this.save(s); await this.broadcast(s, 'game_started'); return json({ room: publicRoom(s), state: viewFor(s, seat) }); }
      if (url.pathname.endsWith('/ws')) return this.acceptWebSocket(request, s, seat);
      return json({ error: 'Not found' }, 404);
    } catch (error) { return json({ error: error instanceof Error ? error.message : 'Bad request' }, 400); }
  }
  private acceptWebSocket(request: Request, s: RoomState, seat: number) {
    if (request.headers.get('Upgrade') !== 'websocket') return new Response('Expected WebSocket', { status: 426 });
    const pair = new WebSocketPair(); const [client, server] = Object.values(pair); server.accept(); this.sockets.set(server, seat); s.players[seat]!.connected = true;
    server.send(JSON.stringify({ type: 'connected', room: publicRoom(s), state: s.status === 'lobby' ? undefined : viewFor(s, seat) }));
    server.addEventListener('message', (event) => void this.onMessage(server, seat, String(event.data)));
    server.addEventListener('close', () => { this.sockets.delete(server); void this.disconnected(seat); });
    return new Response(null, { status: 101, webSocket: client });
  }
  private async disconnected(seat: number) { const s = await this.state(); if (s.players[seat]) { s.players[seat]!.connected = false; await this.save(s); await this.broadcast(s, 'room_updated'); } }
  private async onMessage(socket: WebSocket, seat: number, raw: string) {
    try { const message = JSON.parse(raw) as { type: string; command_id?: string; expected_version?: number; action?: Action | number }; if (message.type === 'ping') return void socket.send(JSON.stringify({ type: 'pong' })); const s = await this.state();
      if (!message.command_id || message.expected_version === undefined) throw new Error('command_id and expected_version are required');
      if (s.processed[message.command_id]) return void socket.send(JSON.stringify(s.processed[message.command_id]));
      if (message.expected_version !== s.stateVersion) { socket.send(JSON.stringify({ type: 'resync_required', room: publicRoom(s), state: viewFor(s, seat) })); return; }
      if (message.type === 'make_action' && message.action !== undefined) { const action: Action = typeof message.action === 'number' ? (message.action === 0 ? { action_type: 'challenge' } : message.action === 1 ? { action_type: 'believe' } : { action_type: 'bid', quantity: Math.floor((message.action - 2) / 6) + 1, value: ((message.action - 2) % 6) + 1 }) : message.action; applyAction(s, seat, action); } else if (message.type === 'continue_round') continueRound(s, seat); else throw new Error('Unknown command');
      s.stateVersion++; const event = { type: s.status === 'finished' ? 'game_finished' : 'state_updated', room: publicRoom(s), state: viewFor(s, seat) }; s.processed[message.command_id] = event; await this.save(s); await this.broadcast(s, event.type); if (s.status === 'finished') await this.persistHistory(s);
    } catch (error) { socket.send(JSON.stringify({ type: 'action_rejected', error: error instanceof Error ? error.message : 'Invalid command' })); }
  }
  private async persistHistory(s: RoomState) { if (!s.gameId) return; const now = Date.now(); const statements = [this.env.DB.prepare('INSERT OR IGNORE INTO games (id, room_id, started_at, finished_at, winner_seat, turns, status, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)').bind(s.gameId, s.roomId, now, now, s.winner, s.history.length, 'finished', now + 2_592_000_000)]; for (const p of s.players) if (p) statements.push(this.env.DB.prepare('INSERT OR IGNORE INTO game_players (game_id, seat, display_name, token_hash, is_winner) VALUES (?, ?, ?, ?, ?)').bind(s.gameId, p.seat, p.name, p.tokenHash, p.seat === s.winner ? 1 : 0)); for (const h of s.history) statements.push(this.env.DB.prepare('INSERT OR IGNORE INTO game_actions (game_id, turn_number, player_seat, type, payload_json, consequences_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)').bind(s.gameId, h.turn_number, h.player_id, h.action_type, JSON.stringify(h.action_data), JSON.stringify(h.consequences), now)); await this.env.DB.batch(statements); }
  async alarm() { const s = await this.state(); if (Date.now() - s.lastActivity >= 86_400_000 && s.status !== 'finished') await this.ctx.storage.deleteAll(); }
}
const cleanName = (name: string) => name.trim().replace(/[<>]/g, '').slice(0, 32) || 'Player';
