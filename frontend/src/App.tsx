import React, { useState, useEffect } from 'react';
import { GameBoard } from './components/GameBoard';
import { gamesApi, GameState } from './services/api';

function App() {
  const [currentGameId, setCurrentGameId] = useState<string | null>(null);
  const [initialGameState, setInitialGameState] = useState<GameState | null>(null);

  useEffect(() => {
    const startGame = async () => {
      try {
        const result = await gamesApi.create({ model_paths: [] });
        setCurrentGameId(result.game_id);
        setInitialGameState(result.state);
      } catch (error) {
        console.error('Failed to start game automatically:', error);
        alert('Failed to start game. Please check the console and try again.');
      }
    };

    startGame();
  }, []);

  const handleGameEnd = () => {
    console.log("Game ended");
    setCurrentGameId(null);
    setInitialGameState(null);
  };

  return (
    <div className="bg-gray-800 min-h-screen text-white p-4 sm:p-6 lg:p-8 flex flex-col items-center font-sans">
      <div className="w-full max-w-7xl text-center mb-6 z-10 relative">
        <h1 className="text-5xl font-bold text-orange-400 mb-2">Secret Tea Society 🍵</h1>
        <p className="text-gray-400 text-lg">The last player with dice wins!</p>
      </div>

      <main className="w-full max-w-7xl">
        {currentGameId && initialGameState ? (
          <GameBoard
            gameId={currentGameId}
            initialState={initialGameState}
            onGameEnd={handleGameEnd}
          />
        ) : (
          <div className="text-center">
            <div className="flex items-center justify-center space-x-3">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-yellow-300"></div>
              <span className="text-2xl font-semibold text-gray-300">Loading Game...</span>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
