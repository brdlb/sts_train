import React, { useState, useEffect, useRef } from 'react';
import { gamesApi, GameState, ExtendedActionHistoryEntry } from '../services/api';
import Player from './Player';
import BidControls from './BidControls';
import { GameHistory } from './GameHistory';
import Modal from './Modal';
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
  const [lastActionHistoryLength, setLastActionHistoryLength] = useState(0);
  const eventSourceRef = useRef<EventSource | null>(null);
  
  // Refs for auto-scrolling
  const playerRefs = useRef<Record<number, HTMLDivElement | null>>({});
  const bidControlsRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    loadGameState();
    // Poll for game state updates (only when not processing AI turns)
    const interval = setInterval(() => {
      if (!gameState?.game_over && !processing && !eventSourceRef.current) {
        loadGameState();
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
  }, [gameId, gameState?.game_over, processing]);

  // Check for new challenge/believe results
  useEffect(() => {
    if (!gameState?.extended_action_history) return;
    
    const currentLength = gameState.extended_action_history.length;
    if (currentLength > lastActionHistoryLength) {
      const newEntries = gameState.extended_action_history.slice(lastActionHistoryLength);
      
      // Check if any new entry is a challenge or believe with consequences
      for (const entry of newEntries) {
        if ((entry.action_type === 'challenge' || entry.action_type === 'believe') && entry.consequences) {
          const consequences = entry.consequences;
          const challengerName = PLAYER_NAMES[entry.player_id] || `Player ${entry.player_id}`;
          const bidderName = consequences.bidder_id !== null && consequences.bidder_id !== undefined
            ? (PLAYER_NAMES[consequences.bidder_id] || `Player ${consequences.bidder_id}`)
            : 'another player';
          
          let title = '';
          let body = '';
          
          if (entry.action_type === 'challenge') {
            title = `${challengerName} challenged the bid!`;
            const bidInfo = consequences.bid_quantity && consequences.bid_value
              ? `${consequences.bid_quantity}x${consequences.bid_value}`
              : 'the bid';
            
            if (consequences.challenge_success === true) {
              body = `The bid ${bidInfo} from ${bidderName} was incorrect! ${bidderName} lost a die.`;
            } else if (consequences.challenge_success === false) {
              body = `The bid ${bidInfo} from ${bidderName} was correct! ${challengerName} lost a die.`;
            } else {
              body = `${challengerName} challenged ${bidInfo} from ${bidderName}.`;
            }
            
            if (consequences.actual_count !== null) {
              body += ` Actual count: ${consequences.actual_count}.`;
            }
          } else if (entry.action_type === 'believe') {
            title = `${challengerName} believed the bid!`;
            const bidInfo = consequences.bid_quantity && consequences.bid_value
              ? `${consequences.bid_quantity}x${consequences.bid_value}`
              : 'the bid';
            
            if (consequences.believe_success === true) {
              body = `The bid ${bidInfo} from ${bidderName} was exact! ${challengerName} gained an advantage.`;
            } else if (consequences.believe_success === false) {
              body = `The bid ${bidInfo} from ${bidderName} was not exact! ${challengerName} lost a die.`;
            } else {
              body = `${challengerName} believed ${bidInfo} from ${bidderName}.`;
            }
            
            if (consequences.actual_count !== null) {
              body += ` Actual count: ${consequences.actual_count}.`;
            }
          }
          
          if (title && body) {
            setGamePhase('round_over');
            setModalContent({ 
              title, 
              body: <p dangerouslySetInnerHTML={{ __html: body.replace(/\n/g, '<br/>') }} /> 
            });
            
            // Auto-close modal after 3 seconds and continue
            setTimeout(() => {
              setModalContent(null);
              setGamePhase('bidding');
            }, 3000);
          }
        }
      }
      
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

  const loadGameState = async () => {
    try {
      const state = await gamesApi.getState(gameId);
      setGameState(state);
      setLoading(false);
      setError(null);
      
      if (state.extended_action_history) {
        setLastActionHistoryLength(state.extended_action_history.length);
      }

      if (state.game_over) {
        setGamePhase('game_over');
        onGameEnd();
      } else if (state.current_player !== 0 && !eventSourceRef.current) {
        // If it's AI's turn and we're not already subscribed, subscribe to AI turns
        subscribeToAiTurns();
      }
    } catch (err) {
      setError('Failed to load game state');
      console.error(err);
      setLoading(false);
    }
  };

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
      setError('Failed to call believe');
      console.error(err);
      setProcessing(false);
      setGamePhase('bidding');
    }
  };

  const subscribeToAiTurns = () => {
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
  
  // Find last bidder from bid history
  const lastBidderId = gameState.bid_history.length > 0 
    ? gameState.bid_history[gameState.bid_history.length - 1][0]
    : null;

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

      {gameState.game_over && (
        <div className={`w-full max-w-7xl mb-6 p-4 rounded-lg ${
          gameState.winner === 0 ? 'bg-green-600' : 'bg-red-600'
        } text-white text-center text-xl font-bold`}>
          {gameState.winner === 0
            ? '🎉 You Won! 🎉'
            : `Game Over! ${PLAYER_NAMES[gameState.winner] || `Player ${gameState.winner}`} won.`}
        </div>
      )}

      <div className="w-full max-w-7xl grid grid-cols-1 lg:grid-cols-2 lg:gap-8 lg:items-end z-10">
        {/* Player List Column */}
        <div className="w-full flex flex-col justify-start space-y-4">
          {displayPlayers.map((playerId) => (
            <Player
              ref={el => { playerRefs.current[playerId] = el; }}
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
        <div className="w-full space-y-4">
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

          <div className="flex-1 min-h-0">
            <GameHistory
              bidHistory={gameState.bid_history}
              currentBid={gameState.current_bid}
              extendedActionHistory={gameState.extended_action_history}
            />
          </div>
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
    </div>
  );
};
