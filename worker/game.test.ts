import { describe, expect, it } from 'vitest';
import { applyAction, continueRound, encodeBid, isHigherBid, viewFor } from './game';
import type { RoomState } from './types';

describe('Perudo Cloudflare game core', () => {
  const playingState = (): RoomState => ({
    roomId: 'room', joinCode: 'ABC123', status: 'playing', hostSeat: 0,
    players: [
      { seat: 0, name: 'Alice', tokenHash: 'a', dice: [2, 2], diceCount: 2, connected: true },
      { seat: 1, name: 'Bob', tokenHash: 'b', dice: [3, 3], diceCount: 2, connected: true },
      { seat: 2, name: 'Carol', tokenHash: 'c', dice: [4, 4], diceCount: 2, connected: true },
      null, null, null,
    ],
    gameId: 'game', currentPlayer: 1, currentBid: [3, 6], lastBidPlayer: 0,
    bidHistory: [[0, 3, 6]], history: [], palifico: [false, false, false, false, false, false], specialRound: false,
    round: 1, stateVersion: 1, winner: null, awaitingReveal: false, lastActivity: 0, processed: {},
  });

  it('keeps the existing action encoding contract', () => {
    expect(encodeBid(1, 1)).toBe(0);
    expect(encodeBid(2, 6)).toBe(11);
  });

  it('applies ones bid rules', () => {
    expect(isHigherBid(3, 2, [1, 1])).toBe(true);
    expect(isHigherBid(2, 2, [1, 1])).toBe(false);
    expect(isHigherBid(1, 1, [2, 4])).toBe(true);
  });

  it('only exposes players who joined the room', () => {
    const state: RoomState = {
      roomId: 'room', joinCode: 'ABC123', status: 'playing', hostSeat: 0,
      players: [
        { seat: 0, name: 'Alice', tokenHash: 'a', dice: [1, 2], diceCount: 2, connected: true },
        { seat: 1, name: 'Bob', tokenHash: 'b', dice: [3], diceCount: 1, connected: true },
        null, null, null, null,
      ],
      gameId: 'game', currentPlayer: 0, currentBid: null, lastBidPlayer: null,
      bidHistory: [], history: [], palifico: [false, false, false, false, false, false], specialRound: false,
      round: 1, stateVersion: 1, winner: null, awaitingReveal: false, lastActivity: 0, processed: {},
    };

    expect(viewFor(state, 0).player_ids).toEqual([0, 1]);
  });

  it('starts the next round with the player who lost a die, not the confirmer', () => {
    const state = playingState();

    applyAction(state, 1, { action_type: 'challenge' }, () => 0);
    expect(state.currentPlayer).toBe(0);
    expect(state.players[0]!.diceCount).toBe(1);

    continueRound(state, 2, () => 0);
    expect(state.currentPlayer).toBe(0);
  });

  it('starts the next round with the challenger when the challenge fails', () => {
    const state = playingState();
    state.currentBid = [1, 2];

    applyAction(state, 1, { action_type: 'challenge' }, () => 0);
    expect(state.currentPlayer).toBe(1);
    expect(state.players[1]!.diceCount).toBe(1);

    continueRound(state, 2, () => 0);
    expect(state.currentPlayer).toBe(1);
  });
});
