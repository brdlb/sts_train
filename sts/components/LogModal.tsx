

import React from 'react';
import type { LogEntry, Player } from '../types';
import GameLog from './GameLog';

interface LogModalProps {
  isOpen: boolean;
  onClose: () => void;
  logs: LogEntry[];
  players: Player[];
}

const LogModal: React.FC<LogModalProps> = ({ isOpen, onClose, logs, players }) => {
  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div 
        className="bg-gray-800 rounded-lg shadow-xl w-full h-full flex flex-col"
        onClick={(e) => e.stopPropagation()} // Prevent closing when clicking inside the modal
      >
        <div className="p-4 border-b border-gray-700 flex justify-between items-center flex-shrink-0">
            <h2 className="text-2xl font-bold text-white">Журнал игры</h2>
            <button onClick={onClose} className="text-gray-400 hover:text-white">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
            </button>
        </div>
        <div className="p-4 flex-grow flex flex-col min-h-0">
            <GameLog logs={logs} players={players} />
        </div>
      </div>
    </div>
  );
};

export default LogModal;