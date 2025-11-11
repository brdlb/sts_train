import React, { useState, useEffect } from 'react';
import { statisticsApi, gamesApi, PlayerStatistics, ModelStatistics, GameHistory } from '../services/api';
import { GameHistoryModal } from './GameHistoryModal';

export const Statistics: React.FC = () => {
  const [playerStats, setPlayerStats] = useState<PlayerStatistics | null>(null);
  const [modelStats, setModelStats] = useState<ModelStatistics | null>(null);
  const [games, setGames] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedHistory, setSelectedHistory] = useState<GameHistory | null>(null);
  const [loadingHistory, setLoadingHistory] = useState(false);

  useEffect(() => {
    loadStatistics();
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
    return <div>Loading statistics...</div>;
  }

  return (
    <div style={{ padding: '20px', maxWidth: '1200px', margin: '0 auto' }}>
      <h2>Statistics</h2>

      {playerStats && (
        <div style={{ marginBottom: '30px', border: '1px solid #ccc', padding: '20px', borderRadius: '8px' }}>
          <h3>Your Statistics</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '15px' }}>
            <div>
              <strong>Total Games:</strong> {playerStats.total_games}
            </div>
            <div>
              <strong>Games Won:</strong> {playerStats.games_won}
            </div>
            <div>
              <strong>Winrate:</strong> {(playerStats.winrate * 100).toFixed(1)}%
            </div>
            <div>
              <strong>Average Game Duration:</strong> {Math.round(playerStats.avg_duration_seconds)}s
            </div>
          </div>
        </div>
      )}

      {modelStats && Object.keys(modelStats).length > 0 && (
        <div style={{ marginBottom: '30px', border: '1px solid #ccc', padding: '20px', borderRadius: '8px' }}>
          <h3>Model Statistics</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #ccc' }}>
                <th style={{ padding: '10px', textAlign: 'left' }}>Model</th>
                <th style={{ padding: '10px', textAlign: 'right' }}>Games</th>
                <th style={{ padding: '10px', textAlign: 'right' }}>Wins</th>
                <th style={{ padding: '10px', textAlign: 'right' }}>Winrate</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(modelStats).map(([modelPath, stats]) => (
                <tr key={modelPath} style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '10px' }}>{modelPath}</td>
                  <td style={{ padding: '10px', textAlign: 'right' }}>{stats.games_count}</td>
                  <td style={{ padding: '10px', textAlign: 'right' }}>{stats.wins_count}</td>
                  <td style={{ padding: '10px', textAlign: 'right' }}>
                    {(stats.winrate * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {games.length > 0 && (
        <div style={{ border: '1px solid #ccc', padding: '20px', borderRadius: '8px' }}>
          <h3>Recent Games</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #ccc' }}>
                <th style={{ padding: '10px', textAlign: 'left' }}>Game ID</th>
                <th style={{ padding: '10px', textAlign: 'left' }}>Created</th>
                <th style={{ padding: '10px', textAlign: 'left' }}>Finished</th>
                <th style={{ padding: '10px', textAlign: 'center' }}>Winner</th>
                <th style={{ padding: '10px', textAlign: 'center' }}>Status</th>
                <th style={{ padding: '10px', textAlign: 'center' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {games.map((game) => (
                <tr
                  key={game.id}
                  style={{
                    borderBottom: '1px solid #eee',
                    cursor: game.is_finished ? 'pointer' : 'default',
                    transition: 'background-color 0.2s',
                  }}
                  onMouseEnter={(e) => {
                    if (game.is_finished) {
                      e.currentTarget.style.backgroundColor = '#f5f5f5';
                    }
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = '';
                  }}
                  onClick={() => {
                    if (game.is_finished && !loadingHistory) {
                      handleGameClick(game.id);
                    }
                  }}
                >
                  <td style={{ padding: '10px' }}>{game.id}</td>
                  <td style={{ padding: '10px' }}>
                    {game.created_at ? new Date(game.created_at).toLocaleString() : '-'}
                  </td>
                  <td style={{ padding: '10px' }}>
                    {game.finished_at ? new Date(game.finished_at).toLocaleString() : '-'}
                  </td>
                  <td style={{ padding: '10px', textAlign: 'center' }}>
                    {game.winner !== null ? `Player ${game.winner}` : '-'}
                  </td>
                  <td style={{ padding: '10px', textAlign: 'center' }}>
                    {game.is_finished ? 'Finished' : 'Active'}
                  </td>
                  <td style={{ padding: '10px', textAlign: 'center' }} onClick={(e) => e.stopPropagation()}>
                    <button
                      onClick={() => handleGameClick(game.id)}
                      disabled={loadingHistory || !game.is_finished}
                      style={{
                        padding: '6px 12px',
                        backgroundColor: game.is_finished ? '#2196F3' : '#ccc',
                        color: 'white',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: loadingHistory || !game.is_finished ? 'not-allowed' : 'pointer',
                        opacity: loadingHistory || !game.is_finished ? 0.6 : 1,
                      }}
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

