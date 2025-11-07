import React, { useState, useEffect } from 'react';
import { gamesApi, GameState } from '../services/api';
import { DiceDisplay } from './DiceDisplay';
import { BidInput } from './BidInput';
import { GameHistory } from './GameHistory';
import { encode_bid } from '../utils/actions';

interface GameBoardProps {
  gameId: string;
  onGameEnd: () => void;
}

export const GameBoard: React.FC<GameBoardProps> = ({ gameId, onGameEnd }) => {
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);

  useEffect(() => {
    loadGameState();
    // Poll for game state updates
    const interval = setInterval(() => {
      if (!gameState?.game_over && !processing) {
        loadGameState();
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(interval);
  }, [gameId, gameState?.game_over, processing]);

  const loadGameState = async () => {
    try {
      const state = await gamesApi.getState(gameId);
      setGameState(state);
      setLoading(false);
      setError(null);

      if (state.game_over) {
        onGameEnd();
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

      if (result.game_over) {
        onGameEnd();
      }
    } catch (err) {
      setError('Failed to make bid');
      console.error(err);
    } finally {
      setProcessing(false);
    }
  };

  const handleChallenge = async () => {
    if (!gameState || processing) return;

    try {
      setProcessing(true);
      const result = await gamesApi.makeAction(gameId, 0); // 0 = challenge
      setGameState(result.state);

      if (result.game_over) {
        onGameEnd();
      }
    } catch (err) {
      setError('Failed to challenge');
      console.error(err);
    } finally {
      setProcessing(false);
    }
  };

  const handleBelieve = async () => {
    if (!gameState || processing) return;

    try {
      setProcessing(true);
      const result = await gamesApi.makeAction(gameId, 1); // 1 = believe
      setGameState(result.state);

      if (result.game_over) {
        onGameEnd();
      }
    } catch (err) {
      setError('Failed to call believe');
      console.error(err);
    } finally {
      setProcessing(false);
    }
  };

  if (loading) {
    return <div>Loading game...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  if (!gameState) {
    return <div>Game not found</div>;
  }

  const playerNames = ['You (Human)', 'AI Player 1', 'AI Player 2', 'AI Player 3'];
  // Extract player dice from observation static_info (last 5 values are dice)
  const playerDice: number[] = [];
  if (gameState.player_dice?.static_info) {
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

  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      <h2>Game {gameId}</h2>

      {gameState.game_over && (
        <div
          style={{
            padding: '15px',
            backgroundColor: gameState.winner === 0 ? '#4CAF50' : '#f44336',
            color: 'white',
            borderRadius: '5px',
            marginBottom: '20px',
            fontSize: '18px',
            fontWeight: 'bold',
          }}
        >
          {gameState.winner === 0
            ? '🎉 You Won! 🎉'
            : `Game Over! Player ${gameState.winner} won.`}
        </div>
      )}

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', marginBottom: '20px' }}>
        {gameState.player_dice_count.map((diceCount, index) => (
          <DiceDisplay
            key={index}
            dice={index === 0 ? playerDice : []}
            playerId={index}
            playerName={playerNames[index]}
            isCurrentPlayer={gameState.current_player === index}
            isHuman={index === 0}
            diceCount={diceCount}
          />
        ))}
      </div>

      {gameState.current_player === 0 && !gameState.game_over && (
        <BidInput
          currentBid={gameState.current_bid}
          maxQuantity={30}
          onBid={handleBid}
          onChallenge={handleChallenge}
          onBelieve={handleBelieve}
          canChallenge={canChallenge}
          canBelieve={canBelieve}
          disabled={processing}
        />
      )}

      {gameState.current_player !== 0 && !gameState.game_over && (
        <div
          style={{
            padding: '20px',
            backgroundColor: '#fff3cd',
            borderRadius: '5px',
            marginTop: '20px',
            textAlign: 'center',
          }}
        >
          Waiting for {playerNames[gameState.current_player]} to make a move...
        </div>
      )}

      <GameHistory bidHistory={gameState.bid_history} currentBid={gameState.current_bid} />
    </div>
  );
};

