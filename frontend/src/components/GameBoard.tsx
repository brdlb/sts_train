import React, { useState, useEffect, useRef, useCallback } from 'react';
import { gamesApi, GameState, ExtendedActionHistoryEntry, ActionResult, Room, roomsApi } from '../services/api';
import Player from './Player';
import BidControls from './BidControls';
import { GameHistory } from './GameHistory';
import DiceRevealModal from './DiceRevealModal';
import GameOverModal from './GameOverModal';
import { encode_bid } from '../utils/actions';
import { PLAYER_NAMES } from '../constants';

interface GameBoardProps {
  gameId: string;
  roomId?: string;
  playerToken?: string;
  myPlayerId?: number;
  initialRoom?: Room;
  onGameEnd: () => void;
}

export const GameBoard: React.FC<GameBoardProps> = ({
  gameId,
  roomId,
  playerToken,
  myPlayerId,
  initialRoom,
  onGameEnd,
}) => {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [_room, setRoom] = useState<Room | null>(initialRoom || null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [gamePhase, setGamePhase] = useState<'bidding' | 'reveal' | 'round_over' | 'game_over'>('bidding');
  const [revealModalEntry, setRevealModalEntry] = useState<ExtendedActionHistoryEntry | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const isMountedRef = useRef(true);
  const isMultiplayer = !!roomId && !!playerToken;
  const sendRoomCommand = useCallback((type: 'make_action' | 'continue_round', action?: number) => {
    if (!socketRef.current || !gameState) throw new Error('Room socket is not connected');
    socketRef.current.send(JSON.stringify({ type, command_id: crypto.randomUUID(), expected_version: gameState.state_version ?? 0, ...(action === undefined ? {} : { action }) }));
  }, [gameState]);

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
    if (isMultiplayer) return;

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
            // Don't call onGameEnd() here - let the modal show first
            // onGameEnd will be called when user closes the GameOverModal
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
  }, [gameId, onGameEnd, isMultiplayer]);

  const handleGameOver = useCallback((result: ActionResult) => {
    if (result.game_over) {
      setGamePhase('game_over');
      // onGameEnd(); // Removed to allow modal to show
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
    if (isMultiplayer) return;
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
        // Don't call onGameEnd() here - let the modal show first
        // onGameEnd will be called when user closes the GameOverModal
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
  }, [gameId, onGameEnd, subscribeToAiTurns, processing, isMultiplayer]);

  useEffect(() => {
    if (!isMultiplayer || !roomId || !playerToken) return;

    setLoading(true);
    setError(null);
    const socket = roomsApi.connect(
      roomId,
      playerToken,
      (event) => {
        if (!isMountedRef.current) return;
        setRoom(event.room);

        if (event.state) {
          setGameState(event.state);
          setLoading(false);
          setProcessing(false);

          if (event.state.game_over || event.type === 'game_finished') {
            setGamePhase('game_over');
          } else if (event.state.awaiting_reveal_confirmation) {
            setGamePhase('round_over');
          } else {
            setGamePhase('bidding');
          }
        }

        if (event.type === 'action_rejected') {
          setError(event.error || 'Action rejected');
          setProcessing(false);
          setGamePhase('bidding');
        }
      },
      () => {
        if (!isMountedRef.current) return;
        setError('Connection error');
        setLoading(false);
        setProcessing(false);
      }
    );

    socketRef.current = socket;

    return () => {
      socket.close();
      socketRef.current = null;
    };
  }, [isMultiplayer, roomId, playerToken]);

  useEffect(() => {
    if (isMultiplayer) return;
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
  }, [gameId, loadGameState, gameState?.game_over, processing, gameState?.current_player, isMultiplayer]);


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
      const currentMyPlayerId = gameState.my_player_id ?? myPlayerId ?? 0;
      if (currentPlayer === currentMyPlayerId) {
        bidControlsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      } else {
        const playerElement = playerRefs.current[currentPlayer];
        playerElement?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }, 700);

    return () => clearTimeout(timeoutId);
  }, [gameState?.current_player, gameState?.turn_number, gameState?.my_player_id, myPlayerId]);

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
      if (isMultiplayer && socketRef.current) {
        sendRoomCommand('make_action', action);
        return;
      }
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
      if (isMultiplayer && socketRef.current) {
        sendRoomCommand('make_action', 0);
        return;
      }
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
      if (isMultiplayer && socketRef.current) {
        sendRoomCommand('make_action', 1);
        return;
      }
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
  const activeMyPlayerId = gameState.my_player_id ?? myPlayerId ?? 0;

  // The worker identifies the participants at game start. Fall back to the
  // count array for legacy single-player API responses that do not send it.
  const playerIds = gameState.player_ids ?? gameState.player_dice_count.map((_, playerId) => playerId);
  const displayPlayers = playerIds.filter((playerId) => playerId !== activeMyPlayerId);
  displayPlayers.push(activeMyPlayerId);

  const getPlayerName = (playerId: number) => (
    gameState.player_names?.[playerId] || PLAYER_NAMES[playerId] || `Player ${playerId}`
  );

  return (
    <div className="bg-gray-800 min-h-screen text-white p-4 sm:p-6 lg:p-8 flex flex-col items-center font-sans relative overflow-hidden">
      <div className="w-full max-w-7xl text-center mb-6 z-10 relative">
        <h1 className="text-5xl font-bold text-orange-400 mb-2">Perudo Game</h1>
        <p className="text-gray-400 text-lg">
          Last player with dice wins!{_room ? ` Room ${_room.join_code}` : ''}
        </p>
      </div>


      <div className="w-full max-w-7xl grid grid-cols-1 lg:grid-cols-2 lg:gap-8 lg:items-end z-10">
        <div className="w-full flex flex-col justify-start space-y-4">
          {displayPlayers.map((playerId) => (
            <Player
              ref={setPlayerRef(playerId)}
              key={playerId}
              playerId={playerId}
              playerName={getPlayerName(playerId)}
              dice={playerId === activeMyPlayerId ? playerDice : []}
              diceCount={gameState.player_dice_count[playerId]}
              isCurrent={gameState.current_player === playerId}
              isHuman={playerId === activeMyPlayerId}
              gamePhase={gamePhase}
              lastBid={gameState.current_bid}
              isLastBidder={lastBidderId === playerId}
              bidHistory={gameState.bid_history}
              revealed={gamePhase === 'reveal' || gamePhase === 'round_over'}
            />
          ))}
        </div>

        <div className="w-full space-y-4 flex flex-col">
          {processing && gameState.current_player !== activeMyPlayerId && (
            <div className="bg-gray-700/50 p-4 rounded-lg flex items-center justify-center space-x-3 w-full max-w-lg mx-auto">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-yellow-300"></div>
              <span className="text-xl font-semibold text-yellow-300">
                {getPlayerName(gameState.current_player)} is thinking...
              </span>
            </div>
          )}

          <div ref={bidControlsRef} className="w-full flex justify-center">
            {gameState.current_player === activeMyPlayerId && !gameState.game_over && gamePhase === 'bidding' && (
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
                playerDiceCount={gameState.player_dice_count[activeMyPlayerId]}
              />
            )}
          </div>

          {gameState.current_player !== activeMyPlayerId && !gameState.game_over && gamePhase === 'bidding' && (
            <div className="bg-yellow-500/20 p-4 rounded-lg text-center">
              <p className="text-lg text-yellow-300">
                Waiting for {getPlayerName(gameState.current_player)} to make a move...
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
              if (isMultiplayer && socketRef.current) {
                sendRoomCommand('continue_round');
                setGamePhase('bidding');
                return;
              }
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
        playerNames={gameState?.player_names}
        playerIds={gameState?.player_ids}
      />

      <GameOverModal
        isOpen={gameState?.game_over || false}
        onClose={onGameEnd}
        gameState={gameState}
      />
    </div>
  );
};
