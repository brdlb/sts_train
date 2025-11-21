/**
 * Utility functions for player-related operations
 */

/**
 * Get player name based on player ID
 * @param playerId - The player ID (0 is human, others are AI)
 * @returns Player name string
 */
export const getPlayerName = (playerId: number): string => {
  if (playerId === 0) return 'You (probably Human)';
  return `AI Player ${playerId}`;
};



