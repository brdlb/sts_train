
import React, { useState, useEffect, useRef } from 'react';
import { gamesApi, GameState, Player as PlayerType, Bid } from '../services/api';
import { Player } from './Player';
import { BidControls } from './BidControls';
import { encode_bid } from '../utils/actions';

interface GameBoardProps {
  gameId: string;
  initialState: GameState;
  onGameEnd: () => void;
}

export const GameBoard: React.FC<GameBoardProps> = ({ gameId, initialState, onGameEnd }) => {
  const [gameState, setGameState] = useState<GameState>(initialState);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    const humanPlayer = gameState.players.find(p => p.is_human);
    if (!gameState.game_over && !humanPlayer?.is_current_player) {
      subscribeToAiTurns();
    }
  }, [gameState.game_over, gameState.players]);

  const handlePlaceBid = async (bid: Bid) => {
    if (processing) return;

    try {
      setProcessing(true);
      const action = encode_bid(bid.quantity, bid.face);
      const result = await gamesApi.makeAction(gameId, action);
      setGameState(result.state);

      if (result.game_over) {
        onGameEnd();
      } else {
        subscribeToAiTurns();
      }
    } catch (err) {
      setError('Failed to make bid');
      console.error(err);
      setProcessing(false);
    }
  };

  const handleDudo = async () => {
    if (processing) return;

    try {
      setProcessing(true);
      const result = await gamesApi.makeAction(gameId, 0); // 0 = dudo
      setGameState(result.state);

      if (result.game_over) {
        onGameEnd();
      } else {
        subscribeToAiTurns();
      }
    } catch (err) {
      setError('Failed to call dudo');
      console.error(err);
      setProcessing(false);
    }
  };

  const subscribeToAiTurns = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
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
            eventSourceRef.current = null;
            onGameEnd();
          }
        } else if (data.type === 'done') {
          if (data.state) {
            setGameState(data.state);
          }
          setProcessing(false);
          eventSourceRef.current = null;
        } else if (data.type === 'error') {
          setError(`Error: ${data.error || 'Unknown error'}`);
          setProcessing(false);
          eventSourceRef.current = null;
        }
      },
      (error) => {
        console.error('SSE connection error:', error);
        setError('Connection error while receiving AI turns');
        setProcessing(false);
        eventSourceRef.current = null;
      }
    );

    eventSourceRef.current = eventSource;
  };

  if (loading) {
    return (
      <div className="text-center">
        <div className="flex items-center justify-center space-x-3">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-yellow-300"></div>
          <span className="text-2xl font-semibold text-gray-300">Loading Game...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return <div className="text-center text-red-500">Error: {error}</div>;
  }

  if (!gameState) {
    return <div className="text-center">Game not found</div>;
  }

  const humanPlayer = gameState.players.find(p => p.is_human);
  const isHumanTurn = humanPlayer?.is_current_player ?? false;

  return (
    <div className="w-full max-w-7xl grid grid-cols-1 lg:grid-cols-2 lg:gap-8 lg:items-end z-10">
      <div className="w-full flex flex-col justify-start space-y-4">
        {gameState.players.map(p => (
          <Player
            key={p.id}
            player={p}
            isCurrent={p.is_current_player}
            gamePhase={'BIDDING'}
            lastBid={gameState.current_bid}
            isLastBidder={p.id === gameState.last_bidder_id}
          />
        ))}
      </div>

      <div className="w-full space-y-4">
        {processing && !isHumanTurn && (
          <div className="bg-gray-700/50 p-4 rounded-lg flex items-center justify-center space-x-3 w-full max-w-lg mx-auto">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-yellow-300"></div>
            <span className="text-xl font-semibold text-yellow-300">AI is thinking...</span>
          </div>
        )}
        {isHumanTurn && (
          <BidControls
            currentBid={gameState.current_bid}
            onPlaceBid={handlePlaceBid}
            onDudo={handleDudo}
            totalDiceInPlay={gameState.total_dice_count}
            isDisabled={processing}
          />
        )}
      </div>
    </div>
  );
};
