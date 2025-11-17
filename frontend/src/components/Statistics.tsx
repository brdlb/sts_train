import React, { useState, useEffect } from 'react';
import { statisticsApi, gamesApi, PlayerStatistics, ModelStatistics, GameHistory } from '../services/api';
import { GameHistoryModal } from './GameHistoryModal';
import { useToastContext } from '../contexts/ToastContext';
import { Bar, Line } from 'react-chartjs-2'; // Подключаем графики
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Регистрируем необходимые компоненты Chart.js
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

export const Statistics: React.FC = () => {
  const [playerStats, setPlayerStats] = useState<PlayerStatistics | null>(null);
  const [modelStats, setModelStats] = useState<ModelStatistics | null>(null);
  const [games, setGames] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedHistory, setSelectedHistory] = useState<GameHistory | null>(null);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [chartKey, setChartKey] = useState<number>(0);
  const { showToast } = useToastContext();

  useEffect(() => {
    loadStatistics();
    
    // Cleanup function to reset chart key on unmount
    return () => {
      setChartKey(prev => prev + 1);
    };
  }, []);

  const loadStatistics = async () => {
    try {
      setLoading(true);
      const [player, models, gamesList] = await Promise.all([
        statisticsApi.getPlayer(),
        statisticsApi.getModels(),
        gamesApi.list({ limit: 10 }),
      ]);
      setPlayerStats(player);
      setModelStats(models);
      setGames(gamesList);
    } catch (err) {
      console.error('Failed to load statistics:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleGameClick = async (dbGameId: number) => {
    try {
      setLoadingHistory(true);
      const history = await gamesApi.getHistoryByDbId(dbGameId);
      setSelectedHistory(history);
    } catch (err) {
      console.error('Failed to load game history:', err);
      showToast('Не удалось загрузить историю игры', 'error');
    } finally {
      setLoadingHistory(false);
    }
  };

  const handleCloseModal = () => {
    setSelectedHistory(null);
  };

  if (loading) {
    return (
      <div className="text-center text-lg text-gray-500 p-8">
        Loading statistics...
      </div>
    );
  }

  return (
    <div className="p-5 max-w-6xl mx-auto">
      <h2 className="text-3xl font-bold text-white mb-6">Statistics</h2>

      {playerStats && (
        <div className="mb-8 border border-gray-600 p-5 rounded-lg bg-gray-700/50">
          <h3 className="text-xl font-semibold text-white mb-4">Your Statistics</h3>
          <Bar
            key={`player-stats-chart-${chartKey}`}
            data={{
              labels: ['Total Games', 'Games Won', 'Winrate (%)'],
              datasets: [
                {
                  label: 'Player Stats',
                  data: [
                    playerStats.total_games,
                    playerStats.games_won,
                    parseFloat((playerStats.winrate * 100).toFixed(1)),
                  ],
                  backgroundColor: ['#4CAF50', '#2196F3', '#FFC107'],
                },
              ],
            }}
            options={{
              responsive: true,
              maintainAspectRatio: true,
              plugins: {
                legend: {
                  display: true,
                  position: 'top' as const,
                },
                title: {
                  display: true,
                  text: 'Player Statistics',
                },
              },
              scales: {
                y: {
                  beginAtZero: true,
                },
              },
            }}
          />
        </div>
      )}

      {modelStats && Object.keys(modelStats).length > 0 && (
        <div className="mb-8 border border-gray-600 p-5 rounded-lg bg-gray-700/50">
          <h3 className="text-xl font-semibold text-white mb-4">Model Statistics</h3>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  <th className="p-2.5 text-left border-b-2 border-gray-500 text-white">Model</th>
                  <th className="p-2.5 text-left border-b-2 border-gray-500 text-white">Games</th>
                  <th className="p-2.5 text-left border-b-2 border-gray-500 text-white">Wins</th>
                  <th className="p-2.5 text-left border-b-2 border-gray-500 text-white">Winrate</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(modelStats).map(([modelPath, stats]) => (
                  <tr key={modelPath} className="hover:bg-gray-600/50">
                    <td className="p-2.5 text-gray-200">{modelPath}</td>
                    <td className="p-2.5 text-gray-200">{stats.games_count}</td>
                    <td className="p-2.5 text-gray-200">{stats.wins_count}</td>
                    <td className="p-2.5 text-gray-200">{(stats.winrate * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {games.length > 0 && (
        <div className="mb-8 border border-gray-600 p-5 rounded-lg bg-gray-700/50">
          <h3 className="text-xl font-semibold text-white mb-4">Recent Games</h3>
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr>
                  <th className="p-2.5 text-left border-b-2 border-gray-500 text-white">Created</th>
                  <th className="p-2.5 text-left border-b-2 border-gray-500 text-white">Finished</th>
                  <th className="p-2.5 text-left border-b-2 border-gray-500 text-white">Winner</th>
                  <th className="p-2.5 text-left border-b-2 border-gray-500 text-white">Status</th>
                  <th className="p-2.5 text-left border-b-2 border-gray-500 text-white">Actions</th>
                </tr>
              </thead>
              <tbody>
                {games.map((game) => (
                  <tr
                    key={game.id}
                    onClick={() => {
                      if (game.is_finished && !loadingHistory) {
                        handleGameClick(game.id);
                      }
                    }}
                    className="hover:bg-gray-600/50 cursor-pointer"
                  >
                    <td className="p-2.5 text-gray-200">{game.created_at ? new Date(game.created_at).toLocaleString() : '-'}</td>
                    <td className="p-2.5 text-gray-200">{game.finished_at ? new Date(game.finished_at).toLocaleString() : '-'}</td>
                    <td className="p-2.5 text-gray-200">{game.winner !== null ? `Player ${game.winner}` : '-'}</td>
                    <td className="p-2.5 text-gray-200">{game.is_finished ? 'Finished' : 'Active'}</td>
                    <td className="p-2.5">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleGameClick(game.id);
                        }}
                        disabled={loadingHistory || !game.is_finished}
                        className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded transition-colors"
                      >
                        {loadingHistory ? 'Загрузка...' : 'История'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {selectedHistory && (
        <GameHistoryModal history={selectedHistory} onClose={handleCloseModal} />
      )}
    </div>
  );
};

