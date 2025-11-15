import React, { useState, useEffect } from 'react';
import { statisticsApi, gamesApi, PlayerStatistics, ModelStatistics, GameHistory } from '../services/api';
import { GameHistoryModal } from './GameHistoryModal';
import './Statistics.css'; // Добавляем CSS для стилизации
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
      alert('Не удалось загрузить историю игры');
    } finally {
      setLoadingHistory(false);
    }
  };

  const handleCloseModal = () => {
    setSelectedHistory(null);
  };

  if (loading) {
    return <div className="loading">Loading statistics...</div>;
  }

  return (
    <div className="statistics">
      <h2>Statistics</h2>

      {playerStats && (
        <div className="player-stats">
          <h3>Your Statistics</h3>
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
        <div className="model-stats">
          <h3>Model Statistics</h3>
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Games</th>
                <th>Wins</th>
                <th>Winrate</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(modelStats).map(([modelPath, stats]) => (
                <tr key={modelPath}>
                  <td>{modelPath}</td>
                  <td>{stats.games_count}</td>
                  <td>{stats.wins_count}</td>
                  <td>{(stats.winrate * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {games.length > 0 && (
        <div className="recent-games">
          <h3>Recent Games</h3>
          <table>
            <thead>
              <tr>
                <th>Created</th>
                <th>Finished</th>
                <th>Winner</th>
                <th>Status</th>
                <th>Actions</th>
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
                >
                  <td>{game.created_at ? new Date(game.created_at).toLocaleString() : '-'}</td>
                  <td>{game.finished_at ? new Date(game.finished_at).toLocaleString() : '-'}</td>
                  <td>{game.winner !== null ? `Player ${game.winner}` : '-'}</td>
                  <td>{game.is_finished ? 'Finished' : 'Active'}</td>
                  <td>
                    <button
                      onClick={() => handleGameClick(game.id)}
                      disabled={loadingHistory || !game.is_finished}
                    >
                      {loadingHistory ? 'Загрузка...' : 'История'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {selectedHistory && (
        <GameHistoryModal history={selectedHistory} onClose={handleCloseModal} />
      )}
    </div>
  );
};

