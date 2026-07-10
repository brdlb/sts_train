import { MAX_PLAYERS, MIN_PLAYERS } from './types';
import type { Action, HistoryEntry, RoomState } from './types';

export type Rng = () => number;
export const cryptoRng: Rng = () => crypto.getRandomValues(new Uint32Array(1))[0] / 2 ** 32;
const roll = (count: number, rng: Rng) => Array.from({ length: count }, () => Math.floor(rng() * 6) + 1);
const active = (s: RoomState) => s.players.filter((p) => p && p.diceCount > 0).map((p) => p!.seat);
const nextActive = (s: RoomState, seat: number) => { for (let i = 1; i <= s.players.length; i++) { const n = (seat + i) % s.players.length; if (s.players[n] && s.players[n]!.diceCount > 0) return n; } return seat; };
export const encodeBid = (q: number, v: number) => (q - 1) * 6 + v - 1;
export const isHigherBid = (q: number, v: number, old: [number, number]) => {
  const [oq, ov] = old;
  if (ov === 1) return v === 1 ? q > oq : q >= 2 * oq + 1;
  if (v === 1) return q >= Math.ceil(oq / 2);
  return q > oq || (q === oq && v > ov);
};
export const startGame = (s: RoomState, rng: Rng = cryptoRng) => {
  const occupied = s.players.filter(Boolean).length;
  if (s.status !== 'lobby' || occupied < MIN_PLAYERS || occupied > MAX_PLAYERS) throw new Error(`Room must have ${MIN_PLAYERS}-${MAX_PLAYERS} players`);
  const playerSeats = s.players.flatMap((p) => p ? [p.seat] : []);
  s.status = 'playing'; s.gameId = crypto.randomUUID(); s.currentPlayer = playerSeats[Math.floor(rng() * playerSeats.length)]; s.round = 1;
  s.players.forEach((p) => { if (p) { p.diceCount = 5; p.dice = roll(5, rng); } });
  s.palifico = s.players.map(() => false);
};
const countBid = (s: RoomState, value: number) => s.players.reduce((n, p) => n + (p ? p.dice.filter((d) => d === value || (!s.specialRound && value !== 1 && d === 1)).length : 0), 0);
const append = (s: RoomState, player: number, action: Action, consequences: Record<string, unknown>) => {
  s.history.push({ player_id: player, action_type: action.action_type, action_data: { action_type: action.action_type, quantity: action.action_type === 'bid' ? action.quantity : null, value: action.action_type === 'bid' ? action.value : null }, consequences, turn_number: s.history.length });
};
const beginRound = (s: RoomState, first: number, rng: Rng) => {
  if (active(s).length <= 1) { s.status = 'finished'; s.winner = active(s)[0] ?? null; return; }
  s.round++; s.currentBid = null; s.lastBidPlayer = null; s.specialRound = false; s.awaitingReveal = false; s.currentPlayer = first;
  s.players.forEach((p, i) => { if (p && p.diceCount) p.dice = roll(p.diceCount, rng); s.palifico[i] = !!p && p.diceCount === 1; });
};
export const applyAction = (s: RoomState, seat: number, action: Action, rng: Rng = cryptoRng) => {
  if (s.status !== 'playing') throw new Error('Game is not active');
  if (s.currentPlayer !== seat) throw new Error('Not your turn');
  if (action.action_type === 'bid') {
    const total = s.players.reduce((n, p) => n + (p?.diceCount ?? 0), 0);
    if (!Number.isInteger(action.quantity) || !Number.isInteger(action.value) || action.quantity < 1 || action.quantity > total || action.value < 1 || action.value > 6) throw new Error('Invalid bid');
    if (s.currentBid && !isHigherBid(action.quantity, action.value, s.currentBid)) throw new Error('Bid must be higher');
    if ((s.specialRound || s.palifico[seat]) && s.currentBid && action.value !== s.currentBid[1]) throw new Error('This round does not allow changing value');
    if (!s.currentBid && !s.specialRound && action.value === 1) throw new Error('The first bid cannot be ones');
    s.currentBid = [action.quantity, action.value]; s.lastBidPlayer = seat; s.bidHistory.push([seat, action.quantity, action.value]); append(s, seat, action, {}); s.currentPlayer = nextActive(s, seat); return;
  }
  if (!s.currentBid || s.lastBidPlayer === null) throw new Error('No bid to resolve');
  const [quantity, value] = s.currentBid; const actual = countBid(s, value); const exact = actual === quantity;
  const loser = action.action_type === 'challenge' ? (actual < quantity ? s.lastBidPlayer : seat) : (exact ? null : seat);
  if (loser !== null) s.players[loser]!.diceCount--;
  if (action.action_type === 'believe' && exact && s.players[seat]!.diceCount < 5) s.players[seat]!.diceCount++;
  const allDice = s.players.map((p) => p?.dice ?? []); append(s, seat, action, { actual_count: actual, bid_quantity: quantity, bid_value: value, loser_id: loser, dice_lost: loser === null ? 0 : 1, challenge_success: action.action_type === 'challenge' ? actual < quantity : null, believe_success: action.action_type === 'believe' ? exact : null, all_player_dice: allDice });
  s.awaitingReveal = true;
  if (active(s).length <= 1) { s.status = 'finished'; s.winner = active(s)[0] ?? null; }
};
export const continueRound = (s: RoomState, seat: number, rng: Rng = cryptoRng) => { if (!s.awaitingReveal) throw new Error('No round to continue'); beginRound(s, seat, rng); };
export const viewFor = (s: RoomState, seat: number) => ({ game_id: s.gameId, my_player_id: seat, current_player: s.currentPlayer, turn_number: s.history.length, game_over: s.status === 'finished', winner: s.winner, player_dice_count: s.players.map((p) => p?.diceCount ?? 0), current_bid: s.currentBid, bid_history: s.bidHistory, extended_action_history: s.history, palifico_active: s.palifico, believe_called: false, last_bid_player_id: s.lastBidPlayer, awaiting_reveal_confirmation: s.awaitingReveal, state_version: s.stateVersion, player_names: Object.fromEntries(s.players.filter(Boolean).map((p) => [p!.seat, p!.name])), player_dice: { bid_history: [], static_info: [], dice_values: s.players[seat]?.dice ?? [] } });
