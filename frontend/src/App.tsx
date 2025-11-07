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
    <div>
      <nav
        style={{
          backgroundColor: '#333',
          color: 'white',
          padding: '15px 20px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <h1 style={{ margin: 0 }}>Perudo Game</h1>
        <div style={{ display: 'flex', gap: '15px' }}>
          <button
            onClick={() => setCurrentView('select')}
            style={{
              padding: '8px 16px',
              backgroundColor: currentView === 'select' ? '#4CAF50' : '#555',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
            }}
          >
            New Game
          </button>
          <button
            onClick={() => setCurrentView('statistics')}
            style={{
              padding: '8px 16px',
              backgroundColor: currentView === 'statistics' ? '#4CAF50' : '#555',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
            }}
          >
            Statistics
          </button>
        </div>
      </nav>

      <main style={{ padding: '20px' }}>
        {currentView === 'select' && <ModelSelector onStart={handleStartGame} />}
        {currentView === 'game' && currentGameId && (
          <GameBoard gameId={currentGameId} onGameEnd={handleGameEnd} />
        )}
        {currentView === 'statistics' && <Statistics />}
      </main>
    </div>
  );
}

export default App;

