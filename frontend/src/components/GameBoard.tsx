import React, { useState, useEffect, useRef, useCallback } from 'react';
import { gamesApi, GameState, ExtendedActionHistoryEntry, ActionResult } from '../services/api';
import Player from './Player';
import BidControls from './BidControls';
import { GameHistory } from './GameHistory';
import DiceRevealModal from './DiceRevealModal';
import GameOverModal from './GameOverModal';
import { encode_bid } from '../utils/actions';
import { PLAYER_NAMES } from '../constants';

interface GameBoardProps {
  gameId: string;
  onGameEnd: () => void;
}

export const GameBoard: React.FC<GameBoardProps> = ({ gameId, onGameEnd }) => {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [gamePhase, setGamePhase] = useState<'bidding' | 'reveal' | 'round_over' | 'game_over'>('bidding');
  const [revealModalEntry, setRevealModalEntry] = useState<ExtendedActionHistoryEntry | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const isMountedRef = useRef(true);

  const playerRefs = useRef<Record<number, HTMLDivElement | null>>({});
  const bidControlsRef = useRef<HTMLDivElement | null>(null);

  const findLastRevealEntry = useCallback((history: ExtendedActionHistoryEntry[]): ExtendedActionHistoryEntry | null => {
    for (let i = history.length - 1; i >= 0; i--) {
      const entry = history[i];
      if ((entry.action_type === 'challenge' || entry.action_type === 'believe') &&
        entry.consequences &&
        entry.consequences.all_player_dice) {
        return entry;
      }
    }
    return null;
  }, []);

  const findLastBidderId = useCallback((gameState: GameState): number | null => {
    if (!gameState.current_bid) return null;

    if (gameState.last_bid_player_id !== undefined && gameState.last_bid_player_id !== null) {
      return gameState.last_bid_player_id;
    }

    if (gameState.extended_action_history) {
      for (let i = gameState.extended_action_history.length - 1; i >= 0; i--) {
        const entry = gameState.extended_action_history[i];
        if (entry.action_type === 'bid' &&
          entry.action_data?.quantity === gameState.current_bid[0] &&
          entry.action_data?.value === gameState.current_bid[1]) {
          return entry.player_id;
        }
      }
    }

    if (gameState.bid_history.length > 0) {
      const lastBidInHistory = gameState.bid_history[gameState.bid_history.length - 1];
      if (lastBidInHistory && lastBidInHistory.length >= 3 &&
        lastBidInHistory[1] === gameState.current_bid[0] &&
        lastBidInHistory[2] === gameState.current_bid[1]) {
        return lastBidInHistory[0];
      }
    }

    return null;
  }, []);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  const subscribeToAiTurns = useCallback(() => {
    // Close any existing connection
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    setProcessing(true);
    const eventSource = gamesApi.subscribeToAiTurns(
      gameId,
      (data) => {
        if (data.type === 'ai_turn') {
          if (data.state) {
            setGameState(data.state);
          }

          if (data.game_over) {
            setProcessing(false);
            setGamePhase('game_over');
            eventSourceRef.current = null;
            if (data.winner !== undefined) {
              onGameEnd();
            }
          }
        } else if (data.type === 'done') {
          if (data.state) {
            setGameState(data.state);
          }
          setProcessing(false);
          setGamePhase('bidding');
          eventSourceRef.current = null;
        } else if (data.type === 'error') {
          setError(`Error: ${data.error || 'Unknown error'}`);
          setProcessing(false);
          setGamePhase('bidding');
          eventSourceRef.current = null;
        }
      },
      (error) => {
        console.error('SSE connection error:', error);
        setError('Connection error while receiving AI turns');
        setProcessing(false);
        setGamePhase('bidding');
        eventSourceRef.current = null;
      }
    );

    eventSourceRef.current = eventSource;
  }, [gameId, onGameEnd]);

  const handleGameOver = useCallback((result: ActionResult) => {
    if (result.game_over) {
      setGamePhase('game_over');
      onGameEnd();
    }
  }, [onGameEnd]);

  const handleAfterAction = useCallback((result: ActionResult) => {
    setGameState(result.state);
    handleGameOver(result);
    if (!result.game_over) {
      subscribeToAiTurns();
    }
  }, [handleGameOver, subscribeToAiTurns]);

  const loadGameState = useCallback(async (skipSSECheck: boolean = false) => {
    if (!isMountedRef.current) return;

    if (!skipSSECheck && (eventSourceRef.current || processing)) {
      return;
    }

    try {
      const state = await gamesApi.getState(gameId);
      if (!isMountedRef.current) return;

      setGameState(state);
      setLoading(false);
      setError(null);

      if (state.game_over) {
        setGamePhase('game_over');
        onGameEnd();
      } else if (state.current_player === 0) {
        setProcessing(false);
      } else if (state.current_player !== 0 && !eventSourceRef.current) {
        subscribeToAiTurns();
      }
    } catch (err) {
      if (!isMountedRef.current) return;
      setError('Failed to load game state');
      console.error(err);
      setLoading(false);
    }
  }, [gameId, onGameEnd, subscribeToAiTurns, processing]);

  useEffect(() => {
    if (!isMountedRef.current) return;

    loadGameState(true);
    const interval = setInterval(() => {
      if (
        isMountedRef.current &&
        !gameState?.game_over &&
        !processing &&
        !eventSourceRef.current &&
        gameState?.current_player !== 0
      ) {
        loadGameState(false);
      }
    }, 2000);

    return () => {
      clearInterval(interval);
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, [gameId, loadGameState, gameState?.game_over, processing, gameState?.current_player]);


  const lastAcknowledgedRevealTurnRef = useRef<number | null>(null);

  useEffect(() => {
    if (!gameState?.extended_action_history) return;

    if (revealModalEntry) {
      return;
    }

    const isAwaitingReveal = gameState.awaiting_reveal_confirmation === true;
    const history = gameState.extended_action_history;

    if (isAwaitingReveal) {
      const lastRevealEntry = findLastRevealEntry(history);
      if (lastRevealEntry) {
        // Check if we already acknowledged this specific reveal
        if (lastAcknowledgedRevealTurnRef.current === lastRevealEntry.turn_number) {
          return;
        }

        console.log('Round end detected - showing reveal modal', {
          awaitingReveal: isAwaitingReveal,
          actionType: lastRevealEntry.action_type,
          turnNumber: lastRevealEntry.turn_number
        });
        setGamePhase('round_over');
        setRevealModalEntry(lastRevealEntry);
      } else {
        console.warn('Round end detected but no reveal entry found', {
          awaitingReveal: isAwaitingReveal,
          historyLength: history.length,
        });
      }
    }
  }, [gameState, revealModalEntry, findLastRevealEntry]);

  useEffect(() => {
    if (!gameState) return;

    const timeoutId = setTimeout(() => {
      const currentPlayer = gameState.current_player;
      if (currentPlayer === 0) {
        bidControlsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      } else {
        const playerElement = playerRefs.current[currentPlayer];
        playerElement?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }, 700);

    return () => clearTimeout(timeoutId);
  }, [gameState?.current_player, gameState?.turn_number]);

  const setPlayerRef = useCallback((playerId: number) => {
    return (el: HTMLDivElement | null) => {
      playerRefs.current[playerId] = el;
    };
  }, []);

  const handleBid = async (quantity: number, value: number) => {
    if (!gameState || processing) return;

    try {
      setProcessing(true);
      const action = encode_bid(quantity, value);
      const result = await gamesApi.makeAction(gameId, action);
      handleAfterAction(result);
    } catch (err) {
      setError('Failed to make bid');
      console.error(err);
      setProcessing(false);
    }
  };

  const handleChallenge = async () => {
    if (!gameState || processing) return;

    try {
      setProcessing(true);
      setGamePhase('reveal');
      const result = await gamesApi.makeAction(gameId, 0);
      handleAfterAction(result);
    } catch (err) {
      setError('Failed to challenge');
      console.error(err);
      setProcessing(false);
      setGamePhase('bidding');
    }
  };

  const handleBelieve = async () => {
    if (!gameState || processing) return;

    try {
      setProcessing(true);
      setGamePhase('reveal');
      const result = await gamesApi.makeAction(gameId, 1);
      handleAfterAction(result);
    } catch (err) {
      setError('Failed to call believe');
      console.error(err);
      setProcessing(false);
      setGamePhase('bidding');
    }
  };


  if (loading) {
    return (
      <div className="bg-gray-800 min-h-screen text-white p-4 sm:p-6 lg:p-8 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-yellow-300 mx-auto mb-4"></div>
          <span className="text-2xl font-semibold text-gray-300">Loading game...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 min-h-screen text-white p-4 sm:p-6 lg:p-8 flex items-center justify-center">
        <div className="text-center text-red-400">
          <p className="text-2xl font-semibold">Error: {error}</p>
        </div>
      </div>
    );
  }

  if (!gameState) {
    return (
      <div className="bg-gray-800 min-h-screen text-white p-4 sm:p-6 lg:p-8 flex items-center justify-center">
        <div className="text-center">
          <p className="text-2xl font-semibold text-gray-300">Game not found</p>
        </div>
      </div>
    );
  }

  const playerDice: number[] = [];
  if (gameState.player_dice?.dice_values) {
    playerDice.push(...gameState.player_dice.dice_values);
  } else if (gameState.player_dice?.static_info) {
    const staticInfo = gameState.player_dice.static_info;
    const diceStart = staticInfo.length - 5;
    for (let i = diceStart; i < staticInfo.length; i++) {
      const dieValue = Math.round(staticInfo[i]);
      if (dieValue > 0 && dieValue <= 6) {
        playerDice.push(dieValue);
      }
    }
  }

  const canChallenge = gameState.current_bid !== null && gameState.bid_history.length > 0;
  const canBelieve = gameState.current_bid !== null && !gameState.believe_called;
  const totalDiceInPlay = gameState.player_dice_count.reduce((sum, count) => sum + count, 0);
  const lastBidderId = findLastBidderId(gameState);

  const displayPlayers = [];
  for (let i = 1; i < gameState.player_dice_count.length; i++) {
    displayPlayers.push(i);
  }
  displayPlayers.push(0);

  return (
    <div className="bg-gray-800 min-h-screen text-white p-4 sm:p-6 lg:p-8 flex flex-col items-center font-sans relative overflow-hidden">
      <div className="w-full max-w-7xl text-center mb-6 z-10 relative">
        <h1 className="text-5xl font-bold text-orange-400 mb-2">Perudo Game</h1>
        <p className="text-gray-400 text-lg">Last player with dice wins!</p>
      </div>


      <div className="w-full max-w-7xl grid grid-cols-1 lg:grid-cols-2 lg:gap-8 lg:items-end z-10">
        <div className="w-full flex flex-col justify-start space-y-4">
          {displayPlayers.map((playerId) => (
            <Player
              ref={setPlayerRef(playerId)}
              key={playerId}
              playerId={playerId}
              playerName={PLAYER_NAMES[playerId] || `Player ${playerId}`}
              dice={playerId === 0 ? playerDice : []}
              diceCount={gameState.player_dice_count[playerId]}
              isCurrent={gameState.current_player === playerId}
              isHuman={playerId === 0}
              gamePhase={gamePhase}
              lastBid={gameState.current_bid}
              isLastBidder={lastBidderId === playerId}
              bidHistory={gameState.bid_history}
              revealed={gamePhase === 'reveal' || gamePhase === 'round_over'}
            />
          ))}
        </div>

        <div className="w-full space-y-4 flex flex-col">
          {processing && gameState.current_player !== 0 && (
            <div className="bg-gray-700/50 p-4 rounded-lg flex items-center justify-center space-x-3 w-full max-w-lg mx-auto">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-yellow-300"></div>
              <span className="text-xl font-semibold text-yellow-300">
                {PLAYER_NAMES[gameState.current_player] || `Player ${gameState.current_player}`} is thinking...
              </span>
            </div>
          )}

          <div ref={bidControlsRef} className="w-full flex justify-center">
            {gameState.current_player === 0 && !gameState.game_over && gamePhase === 'bidding' && (
              <BidControls
                currentBid={gameState.current_bid}
                maxQuantity={30}
                onBid={handleBid}
                onChallenge={handleChallenge}
                onBelieve={handleBelieve}
                canChallenge={canChallenge}
                canBelieve={canBelieve}
                disabled={processing}
                totalDiceInPlay={totalDiceInPlay}
                playerDiceCount={gameState.player_dice_count[0]}
              />
            )}
          </div>

          {gameState.current_player !== 0 && !gameState.game_over && gamePhase === 'bidding' && (
            <div className="bg-yellow-500/20 p-4 rounded-lg text-center">
              <p className="text-lg text-yellow-300">
                Waiting for {PLAYER_NAMES[gameState.current_player] || `Player ${gameState.current_player}`} to make a move...
              </p>
            </div>
          )}
        </div>
      </div>

      <div className="w-full max-w-7xl mt-4 z-10">
        <div className="flex-1 min-h-0 overflow-hidden">
          <GameHistory
            bidHistory={gameState.bid_history}
            currentBid={gameState.current_bid}
            extendedActionHistory={gameState.extended_action_history}
          />
        </div>
      </div>

      <DiceRevealModal
        isOpen={!!revealModalEntry}
        onClose={async () => {
          // Mark this reveal as acknowledged before closing
          if (revealModalEntry) {
            lastAcknowledgedRevealTurnRef.current = revealModalEntry.turn_number;
          }

          setRevealModalEntry(null);

          if (gameState?.awaiting_reveal_confirmation) {
            try {
              const result = await gamesApi.continueRound(gameId);
              setGameState(result.state);
              setGamePhase('bidding');

              if (result.state.current_player !== 0 && !result.state.game_over) {
                subscribeToAiTurns();
              }
            } catch (err) {
              console.error('Failed to continue round:', err);
              setError('Failed to continue to next round');
              setGamePhase('bidding');
            }
          } else {
            setGamePhase('bidding');
          }
        }}
        actionEntry={revealModalEntry}
        isSpecialRound={gameState?.palifico_active?.some(p => p) || false}
      />

      <GameOverModal
        isOpen={gameState?.game_over || false}
        onClose={onGameEnd}
        gameState={gameState}
      />
    </div>
  );
};
