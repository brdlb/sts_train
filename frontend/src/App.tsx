import { useState } from 'react';
import { GameBoard } from './components/GameBoard';
import { Statistics } from './components/Statistics';
import { ToastProvider, useToastContext } from './contexts/ToastContext';
import { Room } from './services/api';
import { HelpModal } from './components/HelpModal';
import { MultiplayerLobby } from './components/MultiplayerLobby';

type View = 'select' | 'game' | 'statistics';

function AppContent() {
  const [currentView, setCurrentView] = useState<View>('select');
  const [currentGameId, setCurrentGameId] = useState<string | null>(null);
  const [currentRoom, setCurrentRoom] = useState<Room | null>(null);
  const [playerToken, setPlayerToken] = useState<string | null>(null);
  const [myPlayerId, setMyPlayerId] = useState<number | null>(null);
  const [showHelpModal, setShowHelpModal] = useState(false);
  useToastContext();

  const handleRoomGameStart = (room: Room, token: string, playerId: number) => {
    setCurrentRoom(room);
    setCurrentGameId(room.game_id);
    setPlayerToken(token);
    setMyPlayerId(playerId);
    setCurrentView('game');
  };

  const handleGameEnd = () => {
    setCurrentView('select');
    setCurrentGameId(null);
    setCurrentRoom(null);
    setPlayerToken(null);
    setMyPlayerId(null);
  };

  const initialRoomId = new URLSearchParams(window.location.search).get('room');

  return (
    <div className="h-full flex flex-col bg-gray-800">
      <nav className="text-white px-5 py-4 flex justify-between items-center flex-shrink-0">
        <h1 className="m-0 text-2xl font-bold text-orange-400"></h1>
        <div className="flex gap-4">
          <button
            onClick={() => setCurrentView('select')}
            className={`px-4 py-2 rounded-lg transition-colors ${currentView === 'select'
              ? 'bg-green-600 hover:bg-green-700'
              : 'bg-gray-700 hover:bg-gray-600'
              } text-white font-semibold`}
          >
            New Game
          </button>
          <button
            onClick={() => setCurrentView('statistics')}
            className={`px-4 py-2 rounded-lg transition-colors ${currentView === 'statistics'
              ? 'bg-green-600 hover:bg-green-700'
              : 'bg-gray-700 hover:bg-gray-600'
              } text-white font-semibold`}
          >
            Statistics
          </button>
          <button
            onClick={() => setShowHelpModal(true)}
            className="px-4 py-2 rounded-lg transition-colors bg-gray-700 hover:bg-gray-600 text-white font-semibold text-xl"
            title="Помощь"
          >
            ?
          </button>
        </div>
      </nav>

      <main className="flex-1 overflow-auto p-5 bg-gray-800">
        {currentView === 'select' && (
          <MultiplayerLobby
            initialRoomId={initialRoomId}
            onGameStart={handleRoomGameStart}
          />
        )}
        {currentView === 'game' && currentGameId && currentRoom && playerToken && myPlayerId !== null && (
          <GameBoard
            gameId={currentGameId}
            roomId={currentRoom.room_id}
            playerToken={playerToken}
            myPlayerId={myPlayerId}
            initialRoom={currentRoom}
            onGameEnd={handleGameEnd}
          />
        )}
        {currentView === 'statistics' && <Statistics key="statistics" />}
      </main>
      {showHelpModal && <HelpModal onClose={() => setShowHelpModal(false)} />}
    </div>
  );
}

function App() {
  return (
    <ToastProvider>
      <AppContent />
    </ToastProvider>
  );
}

export default App;

