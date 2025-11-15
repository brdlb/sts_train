import React, { useState } from 'react';
import { ModelSelector } from './components/ModelSelector';
import { GameBoard } from './components/GameBoard';
import { Statistics } from './components/Statistics';
import { gamesApi } from './services/api';

type View = 'select' | 'game' | 'statistics';

function App() {
  const [currentView, setCurrentView] = useState<View>('select');
  const [currentGameId, setCurrentGameId] = useState<string | null>(null);

  const handleStartGame = async (modelPaths: string[]) => {
    try {
      const result = await gamesApi.create({ model_paths: modelPaths });
      setCurrentGameId(result.game_id);
      setCurrentView('game');
    } catch (error) {
      console.error('Failed to start game:', error);
      alert('Failed to start game. Please try again.');
    }
  };

  const handleGameEnd = () => {
    setCurrentView('select');
    setCurrentGameId(null);
  };

  return (
    <div className="h-full flex flex-col bg-gray-800">
      <nav className="bg-gray-900 text-white px-5 py-4 flex justify-between items-center flex-shrink-0 border-b border-gray-700">
        <h1 className="m-0 text-2xl font-bold text-orange-400">Perudo Game</h1>
        <div className="flex gap-4">
          <button
            onClick={() => setCurrentView('select')}
            className={`px-4 py-2 rounded-lg transition-colors ${
              currentView === 'select' 
                ? 'bg-green-600 hover:bg-green-700' 
                : 'bg-gray-700 hover:bg-gray-600'
            } text-white font-semibold`}
          >
            New Game
          </button>
          <button
            onClick={() => setCurrentView('statistics')}
            className={`px-4 py-2 rounded-lg transition-colors ${
              currentView === 'statistics' 
                ? 'bg-green-600 hover:bg-green-700' 
                : 'bg-gray-700 hover:bg-gray-600'
            } text-white font-semibold`}
          >
            Statistics
          </button>
        </div>
      </nav>

      <main className="flex-1 overflow-auto p-5 bg-gray-800">
        {currentView === 'select' && <ModelSelector onStart={handleStartGame} />}
        {currentView === 'game' && currentGameId && (
          <GameBoard gameId={currentGameId} onGameEnd={handleGameEnd} />
        )}
        {currentView === 'statistics' && <Statistics key="statistics" />}
      </main>
    </div>
  );
}

export default App;

