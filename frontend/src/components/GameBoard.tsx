import React, { useState, useEffect, useRef, useCallback } from 'react';
import { gamesApi, GameState, ExtendedActionHistoryEntry } from '../services/api';
import Player from './Player';
import BidControls from './BidControls';
import { GameHistory } from './GameHistory';
import Modal from './Modal';
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
  const [modalContent, setModalContent] = useState<{ title: string; body: React.ReactNode } | null>(null);
  const [revealModalEntry, setRevealModalEntry] = useState<ExtendedActionHistoryEntry | null>(null);
  const [lastActionHistoryLength, setLastActionHistoryLength] = useState(0);
  const eventSourceRef = useRef<EventSource | null>(null);
  const isMountedRef = useRef(true);
  
  // Refs for auto-scrolling
  const playerRefs = useRef<Record<number, HTMLDivElement | null>>({});
  const bidControlsRef = useRef<HTMLDivElement | null>(null);

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
          // Update game state with each AI turn
          if (data.state) {
            setGameState(data.state);
            if (data.state.extended_action_history) {
              setLastActionHistoryLength(data.state.extended_action_history.length);
            }
          }

          // Check if game is over
          if (data.game_over) {
            setProcessing(false);
            setGamePhase('game_over');
            eventSourceRef.current = null;
            if (data.winner !== undefined) {
              onGameEnd();
            }
          }
        } else if (data.type === 'done') {
          // All AI turns completed
          if (data.state) {
            setGameState(data.state);
            if (data.state.extended_action_history) {
              setLastActionHistoryLength(data.state.extended_action_history.length);
            }
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

  const loadGameState = useCallback(async (skipSSECheck: boolean = false) => {
    if (!isMountedRef.current) return;
    
    // Do not load state if SSE is active to avoid race conditions and duplicate updates
    // Skip this check for initial load or when explicitly requested
    if (!skipSSECheck && (eventSourceRef.current || processing)) {
      return;
    }
    
    try {
      const state = await gamesApi.getState(gameId);
      if (!isMountedRef.current) return;
      
      // Double-check that SSE didn't start while we were fetching (only if not initial load)
      if (!skipSSECheck && (eventSourceRef.current || processing)) {
        return;
      }
      
      setGameState(state);
      setLoading(false);
      setError(null);
      
      if (state.extended_action_history) {
        setLastActionHistoryLength(state.extended_action_history.length);
      }

      if (state.game_over) {
        setGamePhase('game_over');
        onGameEnd();
      } else if (state.current_player === 0) {
        // If it's human's turn, ensure processing is false to enable controls
        setProcessing(false);
      } else if (state.current_player !== 0 && !eventSourceRef.current) {
        // If it's AI's turn and we're not already subscribed, subscribe to AI turns
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
    
    // Initial load - skip SSE check for first load
    loadGameState(true);
    // Poll for game state updates (only when not processing AI turns and not human's turn)
    // IMPORTANT: Do not poll when processing is true or SSE is active to avoid race conditions
    const interval = setInterval(() => {
      // Check conditions at the time of execution, not when interval was created
      if (
        isMountedRef.current && 
        !gameState?.game_over && 
        !processing && 
        !eventSourceRef.current && 
        gameState?.current_player !== 0
      ) {
        loadGameState(false);
      }
    }, 2000); // Poll every 2 seconds

    return () => {
      clearInterval(interval);
      // Cleanup SSE connection
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, [gameId, loadGameState, gameState?.game_over, processing, gameState?.current_player]);

  // Check for new challenge/believe results
  useEffect(() => {
    if (!gameState?.extended_action_history) return;
    
    const currentLength = gameState.extended_action_history.length;
    if (currentLength > lastActionHistoryLength) {
      const newEntries = gameState.extended_action_history.slice(lastActionHistoryLength);
      
      // Find the LAST challenge/believe entry with all_player_dice (most recent)
      let lastRevealEntry: ExtendedActionHistoryEntry | null = null;
      for (const entry of newEntries) {
        if ((entry.action_type === 'challenge' || entry.action_type === 'believe') && 
            entry.consequences && 
            entry.consequences.all_player_dice) {
          lastRevealEntry = entry;
        }
      }
      
      // Show modal only for the last entry with all_player_dice
      if (lastRevealEntry) {
        setGamePhase('round_over');
        setRevealModalEntry(lastRevealEntry);
      }
      
      // Update lastActionHistoryLength at the end
      setLastActionHistoryLength(currentLength);
    }
  }, [gameState?.extended_action_history, lastActionHistoryLength]);

  // Auto-scroll to current player
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

  // Optimize ref callback to avoid unnecessary re-renders
  // Must be called before any conditional returns to follow Rules of Hooks
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
      setGameState(result.state);
      
      if (result.state.extended_action_history) {
        setLastActionHistoryLength(result.state.extended_action_history.length);
      }

      if (result.game_over) {
        setGamePhase('game_over');
        onGameEnd();
      } else {
        // Subscribe to AI turns stream
        subscribeToAiTurns();
      }
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
      const result = await gamesApi.makeAction(gameId, 0); // 0 = challenge
      setGameState(result.state);
      
      // Don't update lastActionHistoryLength here - let useEffect handle it

      if (result.game_over) {
        setGamePhase('game_over');
        onGameEnd();
      } else {
        // Subscribe to AI turns stream
        subscribeToAiTurns();
      }
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
      const result = await gamesApi.makeAction(gameId, 1); // 1 = believe
      setGameState(result.state);
      
      // Don't update lastActionHistoryLength here - let useEffect handle it

      if (result.game_over) {
        setGamePhase('game_over');
        onGameEnd();
      } else {
        // Subscribe to AI turns stream
        subscribeToAiTurns();
      }
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

  // Extract player dice from observation static_info or dice_values
  const playerDice: number[] = [];
  if (gameState.player_dice?.dice_values) {
    playerDice.push(...gameState.player_dice.dice_values);
  } else if (gameState.player_dice?.static_info) {
    const staticInfo = gameState.player_dice.static_info;
    // Last 5 values in static_info are player dice
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
  
  // Calculate total dice in play
  const totalDiceInPlay = gameState.player_dice_count.reduce((sum, count) => sum + count, 0);
  
  // Find last bidder from extended_action_history or bid_history
  let lastBidderId: number | null = null;

  // First, try to use last_bid_player_id if available (most reliable)
  if (gameState.current_bid && gameState.last_bid_player_id !== undefined && gameState.last_bid_player_id !== null) {
    lastBidderId = gameState.last_bid_player_id;
  }
  // Try to find from extended_action_history
  else if (gameState.current_bid && gameState.extended_action_history) {
    for (let i = gameState.extended_action_history.length - 1; i >= 0; i--) {
      const entry = gameState.extended_action_history[i];
      if (entry.action_type === 'bid' && 
          entry.action_data?.quantity === gameState.current_bid[0] &&
          entry.action_data?.value === gameState.current_bid[1]) {
        lastBidderId = entry.player_id;
        break;
      }
    }
  }

  // Fallback to bid_history if extended_action_history didn't work
  if (lastBidderId === null && gameState.bid_history.length > 0 && gameState.current_bid) {
    const lastBidInHistory = gameState.bid_history[gameState.bid_history.length - 1];
    if (lastBidInHistory && lastBidInHistory.length >= 3 &&
        lastBidInHistory[1] === gameState.current_bid[0] && 
        lastBidInHistory[2] === gameState.current_bid[1]) {
      lastBidderId = lastBidInHistory[0];
    }
  }

  // Arrange players: human player at bottom
  const displayPlayers = [];
  // Add AI players first
  for (let i = 1; i < gameState.player_dice_count.length; i++) {
    displayPlayers.push(i);
  }
  // Add human player last
  displayPlayers.push(0);

  return (
    <div className="bg-gray-800 min-h-screen text-white p-4 sm:p-6 lg:p-8 flex flex-col items-center font-sans relative overflow-hidden">
      <div className="w-full max-w-7xl text-center mb-6 z-10 relative">
        <h1 className="text-5xl font-bold text-orange-400 mb-2">Perudo Game</h1>
        <p className="text-gray-400 text-lg">Last player with dice wins!</p>
      </div>


      <div className="w-full max-w-7xl grid grid-cols-1 lg:grid-cols-2 lg:gap-8 lg:items-end z-10">
        {/* Player List Column */}
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

        {/* Controls Column */}
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

      <Modal
        isOpen={!!modalContent}
        title={modalContent?.title || ''}
        onClose={() => {
          setModalContent(null);
          setGamePhase('bidding');
        }}
      >
        {modalContent?.body}
      </Modal>

      <DiceRevealModal
        isOpen={!!revealModalEntry}
        onClose={async () => {
          setRevealModalEntry(null);
          
          // If game is awaiting reveal confirmation, continue to next round
          if (gameState?.awaiting_reveal_confirmation) {
            try {
              const result = await gamesApi.continueRound(gameId);
              setGameState(result.state);
              setGamePhase('bidding');
              
              // If it's AI's turn, subscribe to AI turns
              if (result.state.current_player !== 0 && !result.state.game_over) {
                subscribeToAiTurns();
              }
            } catch (err) {
              console.error('Failed to continue round:', err);
              setError('Failed to continue to next round');
              // Fallback: just update phase
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
