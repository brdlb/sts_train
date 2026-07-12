/**
 * Utility functions for player-related operations
 */

/**
 * Get a player's display name from the names sent by the server.
 * @param playerId - The player's seat ID
 * @param playerNames - Names keyed by seat ID
 * @returns Player name string
 */
export const getPlayerName = (
  playerId: number,
  playerNames?: Record<number, string>,
): string => {
  return playerNames?.[playerId] || `Player ${playerId + 1}`;
};





