import React, { useEffect, useState } from 'react';
import { statisticsApi } from '../services/api';

export const Statistics: React.FC = () => {
  const [stats, setStats] = useState<{ total_games: number; average_duration_seconds?: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { statisticsApi.getGames().then(setStats).catch(() => setError('Statistics are temporarily unavailable.')); }, []);
  return <div className="p-5 max-w-2xl mx-auto text-white"><h2 className="text-3xl font-bold mb-6">Game statistics</h2>{error && <p className="text-red-300">{error}</p>}{!stats && !error && <p className="text-gray-300">Loading…</p>}{stats && <div className="rounded-lg bg-gray-700/50 p-6 space-y-3"><p className="text-2xl"><span className="text-gray-300">Completed games: </span>{stats.total_games}</p><p className="text-2xl"><span className="text-gray-300">Average duration: </span>{Math.round(stats.average_duration_seconds ?? 0)} s</p><p className="text-sm text-gray-400">These figures are anonymous and contain no player names or game history.</p></div>}</div>;
};
