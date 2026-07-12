import { RoomDurableObject } from './room';
import type { Env } from './types';
export { RoomDurableObject };

const json = (body: unknown, status = 200) => Response.json(body, { status });
const roomStub = (env: Env, roomId: string) => env.ROOMS.get(env.ROOMS.idFromName(roomId));
const forward = (env: Env, roomId: string, request: Request, suffix: string) => roomStub(env, roomId).fetch(new Request(`https://room.internal/${suffix}`, request));

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url); const parts = url.pathname.split('/').filter(Boolean);
    if (url.pathname === '/health') { try { await env.DB.prepare('SELECT 1').first(); return json({ ok: true, worker: 'perudo', d1: 'ok', version: env.VERSION ?? 'v1' }); } catch { return json({ ok: false, worker: 'perudo', d1: 'unavailable' }, 503); } }
    if (request.method === 'GET' && url.pathname === '/favicon.ico') return new Response(null, { status: 204 });
    if (request.method === 'POST' && url.pathname === '/api/rooms') { const body = await request.json<{ player_name?: string }>(); const roomId = crypto.randomUUID(); return roomStub(env, roomId).fetch('https://room.internal/initialize', { method: 'POST', body: JSON.stringify({ room_id: roomId, player_name: body.player_name ?? 'Host' }), headers: { 'content-type': 'application/json' } }); }
    if (parts[0] === 'api' && parts[1] === 'rooms' && parts[2]) { const roomId = parts[2]; if (request.method === 'GET' && parts.length === 3) return forward(env, roomId, request, 'room'); if (request.method === 'POST' && parts[3] === 'join') return forward(env, roomId, request, 'join'); if (request.method === 'POST' && parts[3] === 'start') return forward(env, roomId, request, 'start'); }
    // Forward the original upgrade request. Rebuilding a Request here strips the
    // WebSocket upgrade in production, so the Durable Object never receives it.
    if (parts[0] === 'ws' && parts[1] === 'rooms' && parts[2]) return roomStub(env, parts[2]).fetch(request);
    if (request.method === 'GET' && url.pathname === '/api/statistics/games') { const row = await env.DB.prepare("SELECT COUNT(*) AS total_games, AVG(finished_at - started_at) AS avg_duration_ms FROM games WHERE status = 'finished'").first<{ total_games: number; avg_duration_ms: number | null }>(); return json({ total_games: row?.total_games ?? 0, average_duration_seconds: Math.round((row?.avg_duration_ms ?? 0) / 1000) }); }
    if (request.method === 'GET' && parts[0] === 'api' && parts[1] === 'games' && parts[2] && parts[3] === 'history') { const token = request.headers.get('Authorization')?.replace(/^Bearer\s+/i, ''); if (!token) return json({ error: 'Unauthorized' }, 401); const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(token)))).map((n) => n.toString(16).padStart(2, '0')).join(''); const player = await env.DB.prepare('SELECT 1 FROM game_players WHERE game_id = ? AND token_hash = ?').bind(parts[2], digest).first(); if (!player) return json({ error: 'Forbidden' }, 403); const [game, actions] = await Promise.all([env.DB.prepare('SELECT * FROM games WHERE id = ?').bind(parts[2]).first(), env.DB.prepare('SELECT turn_number, player_seat, type, payload_json, consequences_json, created_at FROM game_actions WHERE game_id = ? ORDER BY turn_number').bind(parts[2]).all()]); return json({ game, actions: actions.results }); }
    return env.ASSETS.fetch(request);
  },
  async scheduled(_controller: ScheduledController, env: Env) { await env.DB.prepare('DELETE FROM games WHERE expires_at < ?').bind(Date.now()).run(); }
} satisfies ExportedHandler<Env>;
