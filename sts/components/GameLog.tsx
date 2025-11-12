

import React, { useRef, useEffect } from 'react';
import type { LogEntry, Player } from '../types';

interface GameLogProps {
  logs: LogEntry[];
  players: Player[];
}

const GameLog: React.FC<GameLogProps> = ({ logs, players }) => {
  const logContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="w-full h-full bg-gray-900/80 rounded-lg p-4 shadow-lg flex flex-col">
      <h3 className="text-xl font-semibold text-yellow-300 border-b border-yellow-300/30 pb-2 mb-2 flex-shrink-0">Журнал игры</h3>
      <div ref={logContainerRef} className="flex-grow overflow-y-auto pr-2">
        <ul className="space-y-3 text-base">
          {logs.map((log, index) => {
            const player = log.playerId ? players.find(p => p.id === log.playerId) : null;
            if (player) {
              return (
                <li key={index}>
                  <div className={`${player.color} px-3 py-2 rounded-lg text-white shadow`}>
                    <span dangerouslySetInnerHTML={{ __html: `<strong>${player.name}:</strong> ${log.message}` }} />
                  </div>
                </li>
              );
            } else {
              return (
                <li key={index} className="text-gray-200" dangerouslySetInnerHTML={{ __html: log.message }} />
              );
            }
          })}
        </ul>
      </div>
    </div>
  );
};

export default GameLog;