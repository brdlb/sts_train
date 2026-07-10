var __defProp = Object.defineProperty;
var __name = (target, value) => __defProp(target, "name", { value, configurable: true });

// worker/types.ts
var MIN_PLAYERS = 2;
var MAX_PLAYERS = 6;

// worker/game.ts
var cryptoRng = /* @__PURE__ */ __name(() => crypto.getRandomValues(new Uint32Array(1))[0] / 2 ** 32, "cryptoRng");
var roll = /* @__PURE__ */ __name((count, rng) => Array.from({ length: count }, () => Math.floor(rng() * 6) + 1), "roll");
var active = /* @__PURE__ */ __name((s) => s.players.filter((p) => p && p.diceCount > 0).map((p) => p.seat), "active");
var nextActive = /* @__PURE__ */ __name((s, seat) => {
  for (let i = 1; i <= s.players.length; i++) {
    const n = (seat + i) % s.players.length;
    if (s.players[n] && s.players[n].diceCount > 0) return n;
  }
  return seat;
}, "nextActive");
var isHigherBid = /* @__PURE__ */ __name((q, v, old) => {
  const [oq, ov] = old;
  if (ov === 1) return v === 1 ? q > oq : q >= 2 * oq + 1;
  if (v === 1) return q >= Math.ceil(oq / 2);
  return q > oq || q === oq && v > ov;
}, "isHigherBid");
var startGame = /* @__PURE__ */ __name((s, rng = cryptoRng) => {
  const occupied = s.players.filter(Boolean).length;
  if (s.status !== "lobby" || occupied < MIN_PLAYERS || occupied > MAX_PLAYERS) throw new Error(`Room must have ${MIN_PLAYERS}-${MAX_PLAYERS} players`);
  const playerSeats = s.players.flatMap((p) => p ? [p.seat] : []);
  s.status = "playing";
  s.gameId = crypto.randomUUID();
  s.currentPlayer = playerSeats[Math.floor(rng() * playerSeats.length)];
  s.round = 1;
  s.players.forEach((p) => {
    if (p) {
      p.diceCount = 5;
      p.dice = roll(5, rng);
    }
  });
  s.palifico = s.players.map(() => false);
}, "startGame");
var countBid = /* @__PURE__ */ __name((s, value) => s.players.reduce((n, p) => n + (p ? p.dice.filter((d) => d === value || !s.specialRound && value !== 1 && d === 1).length : 0), 0), "countBid");
var append = /* @__PURE__ */ __name((s, player, action, consequences) => {
  s.history.push({ player_id: player, action_type: action.action_type, action_data: { action_type: action.action_type, quantity: action.action_type === "bid" ? action.quantity : null, value: action.action_type === "bid" ? action.value : null }, consequences, turn_number: s.history.length });
}, "append");
var beginRound = /* @__PURE__ */ __name((s, first, rng) => {
  if (active(s).length <= 1) {
    s.status = "finished";
    s.winner = active(s)[0] ?? null;
    return;
  }
  s.round++;
  s.currentBid = null;
  s.lastBidPlayer = null;
  s.specialRound = false;
  s.awaitingReveal = false;
  s.currentPlayer = first;
  s.players.forEach((p, i) => {
    if (p && p.diceCount) p.dice = roll(p.diceCount, rng);
    s.palifico[i] = !!p && p.diceCount === 1;
  });
}, "beginRound");
var applyAction = /* @__PURE__ */ __name((s, seat, action, rng = cryptoRng) => {
  if (s.status !== "playing") throw new Error("Game is not active");
  if (s.currentPlayer !== seat) throw new Error("Not your turn");
  if (action.action_type === "bid") {
    const total = s.players.reduce((n, p) => n + (p?.diceCount ?? 0), 0);
    if (!Number.isInteger(action.quantity) || !Number.isInteger(action.value) || action.quantity < 1 || action.quantity > total || action.value < 1 || action.value > 6) throw new Error("Invalid bid");
    if (s.currentBid && !isHigherBid(action.quantity, action.value, s.currentBid)) throw new Error("Bid must be higher");
    if ((s.specialRound || s.palifico[seat]) && s.currentBid && action.value !== s.currentBid[1]) throw new Error("This round does not allow changing value");
    if (!s.currentBid && !s.specialRound && action.value === 1) throw new Error("The first bid cannot be ones");
    s.currentBid = [action.quantity, action.value];
    s.lastBidPlayer = seat;
    s.bidHistory.push([seat, action.quantity, action.value]);
    append(s, seat, action, {});
    s.currentPlayer = nextActive(s, seat);
    return;
  }
  if (!s.currentBid || s.lastBidPlayer === null) throw new Error("No bid to resolve");
  const [quantity, value] = s.currentBid;
  const actual = countBid(s, value);
  const exact = actual === quantity;
  const loser = action.action_type === "challenge" ? actual < quantity ? s.lastBidPlayer : seat : exact ? null : seat;
  if (loser !== null) s.players[loser].diceCount--;
  if (action.action_type === "believe" && exact && s.players[seat].diceCount < 5) s.players[seat].diceCount++;
  const allDice = s.players.map((p) => p?.dice ?? []);
  append(s, seat, action, { actual_count: actual, bid_quantity: quantity, bid_value: value, loser_id: loser, dice_lost: loser === null ? 0 : 1, challenge_success: action.action_type === "challenge" ? actual < quantity : null, believe_success: action.action_type === "believe" ? exact : null, all_player_dice: allDice });
  s.awaitingReveal = true;
  if (active(s).length <= 1) {
    s.status = "finished";
    s.winner = active(s)[0] ?? null;
  }
}, "applyAction");
var continueRound = /* @__PURE__ */ __name((s, seat, rng = cryptoRng) => {
  if (!s.awaitingReveal) throw new Error("No round to continue");
  beginRound(s, seat, rng);
}, "continueRound");
var viewFor = /* @__PURE__ */ __name((s, seat) => ({ game_id: s.gameId, my_player_id: seat, current_player: s.currentPlayer, turn_number: s.history.length, game_over: s.status === "finished", winner: s.winner, player_dice_count: s.players.map((p) => p?.diceCount ?? 0), current_bid: s.currentBid, bid_history: s.bidHistory, extended_action_history: s.history, palifico_active: s.palifico, believe_called: false, last_bid_player_id: s.lastBidPlayer, awaiting_reveal_confirmation: s.awaitingReveal, state_version: s.stateVersion, player_names: Object.fromEntries(s.players.filter(Boolean).map((p) => [p.seat, p.name])), player_dice: { bid_history: [], static_info: [], dice_values: s.players[seat]?.dice ?? [] } }), "viewFor");

// worker/room.ts
var json = /* @__PURE__ */ __name((body, status = 200) => Response.json(body, { status }), "json");
var hash = /* @__PURE__ */ __name(async (value) => Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)))).map((n) => n.toString(16).padStart(2, "0")).join(""), "hash");
var publicRoom = /* @__PURE__ */ __name((s) => ({ room_id: s.roomId, join_code: s.joinCode, status: s.status, game_id: s.gameId, state_version: s.stateVersion, seats: s.players.map((p, player_id) => ({ player_id, seat_type: p ? "human" : "empty", player_name: p?.name ?? null, is_connected: p?.connected ?? false })) }), "publicRoom");
var roomCode = /* @__PURE__ */ __name(() => crypto.randomUUID().replace(/-/g, "").slice(0, 6).toUpperCase(), "roomCode");
var RoomDurableObject = class {
  constructor(ctx, env) {
    this.ctx = ctx;
    this.env = env;
  }
  ctx;
  env;
  static {
    __name(this, "RoomDurableObject");
  }
  sockets = /* @__PURE__ */ new Map();
  async state() {
    const state = await this.ctx.storage.get("state");
    if (!state) throw new Error("Room not initialized");
    return state;
  }
  async save(s) {
    s.lastActivity = Date.now();
    await this.ctx.storage.put("state", s);
    await this.ctx.storage.setAlarm(s.lastActivity + 864e5);
  }
  token(request) {
    const auth = request.headers.get("Authorization");
    return auth?.startsWith("Bearer ") ? auth.slice(7) : new URL(request.url).searchParams.get("player_token");
  }
  async seat(s, raw) {
    if (!raw) return -1;
    const h = await hash(raw);
    return s.players.findIndex((p) => p?.tokenHash === h);
  }
  async broadcast(s, type) {
    for (const [socket, seat] of this.sockets) {
      try {
        socket.send(JSON.stringify({ type, room: publicRoom(s), state: s.status === "lobby" ? void 0 : viewFor(s, seat) }));
      } catch {
        this.sockets.delete(socket);
      }
    }
  }
  async fetch(request) {
    try {
      const url = new URL(request.url);
      const parts = url.pathname.split("/").filter(Boolean);
      let s;
      if (request.method === "POST" && parts.at(-1) === "initialize") {
        const { room_id, player_name } = await request.json();
        const token = crypto.randomUUID() + crypto.randomUUID();
        s = { roomId: room_id, joinCode: roomCode(), status: "lobby", hostSeat: 0, players: [{ seat: 0, name: cleanName(player_name), tokenHash: await hash(token), dice: [], diceCount: 5, connected: false }, ...Array.from({ length: MAX_PLAYERS - 1 }, () => null)], gameId: null, currentPlayer: 0, currentBid: null, lastBidPlayer: null, bidHistory: [], history: [], palifico: Array(MAX_PLAYERS).fill(false), specialRound: false, round: 0, stateVersion: 1, winner: null, awaitingReveal: false, lastActivity: Date.now(), processed: {} };
        await this.save(s);
        return json({ room: publicRoom(s), player_id: 0, player_token: token }, 201);
      }
      s = await this.state();
      if (request.method === "GET" && parts.at(-1) === "room") return json({ room: publicRoom(s) });
      if (request.method === "POST" && parts.at(-1) === "join") {
        const body = await request.json();
        let seat2 = await this.seat(s, body.player_token ?? null);
        let token = body.player_token;
        if (seat2 < 0) {
          seat2 = s.players.findIndex((p) => !p);
          if (seat2 < 0 || s.status !== "lobby") return json({ error: "Room is full or has already started" }, 409);
          token = crypto.randomUUID() + crypto.randomUUID();
          s.players[seat2] = { seat: seat2, name: cleanName(body.player_name), tokenHash: await hash(token), dice: [], diceCount: 5, connected: false };
          s.stateVersion++;
          await this.save(s);
          await this.broadcast(s, "room_updated");
        }
        return json({ room: publicRoom(s), player_id: seat2, player_token: token });
      }
      const seat = await this.seat(s, this.token(request));
      if (seat < 0) return json({ error: "Unauthorized" }, 401);
      if (request.method === "POST" && parts.at(-1) === "start") {
        if (seat !== s.hostSeat) return json({ error: "Only the host can start" }, 403);
        startGame(s);
        s.stateVersion++;
        await this.save(s);
        await this.broadcast(s, "game_started");
        return json({ room: publicRoom(s), state: viewFor(s, seat) });
      }
      if (url.pathname.endsWith("/ws")) return this.acceptWebSocket(request, s, seat);
      return json({ error: "Not found" }, 404);
    } catch (error) {
      return json({ error: error instanceof Error ? error.message : "Bad request" }, 400);
    }
  }
  acceptWebSocket(request, s, seat) {
    if (request.headers.get("Upgrade") !== "websocket") return new Response("Expected WebSocket", { status: 426 });
    const pair = new WebSocketPair();
    const [client, server] = Object.values(pair);
    server.accept();
    this.sockets.set(server, seat);
    s.players[seat].connected = true;
    server.send(JSON.stringify({ type: "connected", room: publicRoom(s), state: s.status === "lobby" ? void 0 : viewFor(s, seat) }));
    server.addEventListener("message", (event) => void this.onMessage(server, seat, String(event.data)));
    server.addEventListener("close", () => {
      this.sockets.delete(server);
      void this.disconnected(seat);
    });
    return new Response(null, { status: 101, webSocket: client });
  }
  async disconnected(seat) {
    const s = await this.state();
    if (s.players[seat]) {
      s.players[seat].connected = false;
      await this.save(s);
      await this.broadcast(s, "room_updated");
    }
  }
  async onMessage(socket, seat, raw) {
    try {
      const message = JSON.parse(raw);
      if (message.type === "ping") return void socket.send(JSON.stringify({ type: "pong" }));
      const s = await this.state();
      if (!message.command_id || message.expected_version === void 0) throw new Error("command_id and expected_version are required");
      if (s.processed[message.command_id]) return void socket.send(JSON.stringify(s.processed[message.command_id]));
      if (message.expected_version !== s.stateVersion) {
        socket.send(JSON.stringify({ type: "resync_required", room: publicRoom(s), state: viewFor(s, seat) }));
        return;
      }
      if (message.type === "make_action" && message.action !== void 0) {
        const action = typeof message.action === "number" ? message.action === 0 ? { action_type: "challenge" } : message.action === 1 ? { action_type: "believe" } : { action_type: "bid", quantity: Math.floor((message.action - 2) / 6) + 1, value: (message.action - 2) % 6 + 1 } : message.action;
        applyAction(s, seat, action);
      } else if (message.type === "continue_round") continueRound(s, seat);
      else throw new Error("Unknown command");
      s.stateVersion++;
      const event = { type: s.status === "finished" ? "game_finished" : "state_updated", room: publicRoom(s), state: viewFor(s, seat) };
      s.processed[message.command_id] = event;
      await this.save(s);
      await this.broadcast(s, event.type);
      if (s.status === "finished") await this.persistHistory(s);
    } catch (error) {
      socket.send(JSON.stringify({ type: "action_rejected", error: error instanceof Error ? error.message : "Invalid command" }));
    }
  }
  async persistHistory(s) {
    if (!s.gameId) return;
    const now = Date.now();
    const statements = [this.env.DB.prepare("INSERT OR IGNORE INTO games (id, room_id, started_at, finished_at, winner_seat, turns, status, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)").bind(s.gameId, s.roomId, now, now, s.winner, s.history.length, "finished", now + 2592e6)];
    for (const p of s.players) if (p) statements.push(this.env.DB.prepare("INSERT OR IGNORE INTO game_players (game_id, seat, display_name, token_hash, is_winner) VALUES (?, ?, ?, ?, ?)").bind(s.gameId, p.seat, p.name, p.tokenHash, p.seat === s.winner ? 1 : 0));
    for (const h of s.history) statements.push(this.env.DB.prepare("INSERT OR IGNORE INTO game_actions (game_id, turn_number, player_seat, type, payload_json, consequences_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)").bind(s.gameId, h.turn_number, h.player_id, h.action_type, JSON.stringify(h.action_data), JSON.stringify(h.consequences), now));
    await this.env.DB.batch(statements);
  }
  async alarm() {
    const s = await this.state();
    if (Date.now() - s.lastActivity >= 864e5 && s.status !== "finished") await this.ctx.storage.deleteAll();
  }
};
var cleanName = /* @__PURE__ */ __name((name) => name.trim().replace(/[<>]/g, "").slice(0, 32) || "Player", "cleanName");

// worker/index.ts
var json2 = /* @__PURE__ */ __name((body, status = 200) => Response.json(body, { status }), "json");
var roomStub = /* @__PURE__ */ __name((env, roomId) => env.ROOMS.get(env.ROOMS.idFromName(roomId)), "roomStub");
var forward = /* @__PURE__ */ __name((env, roomId, request, suffix) => roomStub(env, roomId).fetch(new Request(`https://room.internal/${suffix}`, request)), "forward");
var index_default = {
  async fetch(request, env) {
    const url = new URL(request.url);
    const parts = url.pathname.split("/").filter(Boolean);
    if (url.pathname === "/health") {
      try {
        await env.DB.prepare("SELECT 1").first();
        return json2({ ok: true, worker: "perudo", d1: "ok", version: env.VERSION ?? "v1" });
      } catch {
        return json2({ ok: false, worker: "perudo", d1: "unavailable" }, 503);
      }
    }
    if (request.method === "POST" && url.pathname === "/api/rooms") {
      const body = await request.json();
      const roomId = crypto.randomUUID();
      return roomStub(env, roomId).fetch("https://room.internal/initialize", { method: "POST", body: JSON.stringify({ room_id: roomId, player_name: body.player_name ?? "Host" }), headers: { "content-type": "application/json" } });
    }
    if (parts[0] === "api" && parts[1] === "rooms" && parts[2]) {
      const roomId = parts[2];
      if (request.method === "GET" && parts.length === 3) return forward(env, roomId, request, "room");
      if (request.method === "POST" && parts[3] === "join") return forward(env, roomId, request, "join");
      if (request.method === "POST" && parts[3] === "start") return forward(env, roomId, request, "start");
    }
    if (parts[0] === "ws" && parts[1] === "rooms" && parts[2]) return forward(env, parts[2], request, "ws");
    if (request.method === "GET" && url.pathname === "/api/statistics/games") {
      const row = await env.DB.prepare("SELECT COUNT(*) AS total_games, AVG(finished_at - started_at) AS avg_duration_ms FROM games WHERE status = 'finished'").first();
      return json2({ total_games: row?.total_games ?? 0, average_duration_seconds: Math.round((row?.avg_duration_ms ?? 0) / 1e3) });
    }
    if (request.method === "GET" && parts[0] === "api" && parts[1] === "games" && parts[2] && parts[3] === "history") {
      const token = request.headers.get("Authorization")?.replace(/^Bearer\s+/i, "");
      if (!token) return json2({ error: "Unauthorized" }, 401);
      const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(token)))).map((n) => n.toString(16).padStart(2, "0")).join("");
      const player = await env.DB.prepare("SELECT 1 FROM game_players WHERE game_id = ? AND token_hash = ?").bind(parts[2], digest).first();
      if (!player) return json2({ error: "Forbidden" }, 403);
      const [game, actions] = await Promise.all([env.DB.prepare("SELECT * FROM games WHERE id = ?").bind(parts[2]).first(), env.DB.prepare("SELECT turn_number, player_seat, type, payload_json, consequences_json, created_at FROM game_actions WHERE game_id = ? ORDER BY turn_number").bind(parts[2]).all()]);
      return json2({ game, actions: actions.results });
    }
    return env.ASSETS.fetch(request);
  },
  async scheduled(_controller, env) {
    await env.DB.prepare("DELETE FROM games WHERE expires_at < ?").bind(Date.now()).run();
  }
};
export {
  RoomDurableObject,
  index_default as default
};
//# sourceMappingURL=index.js.map
