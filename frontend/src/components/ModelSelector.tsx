import React, { useState, useEffect } from 'react';
import { modelsApi, ModelInfo } from '../services/api';

interface ModelSelectorProps {
  onStart: (modelPaths: string[]) => void;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({ onStart }) => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>(['', '', '']);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      setLoading(true);
      const modelList = await modelsApi.list();
      setModels(modelList);
      setError(null);
    } catch (err) {
      setError('Failed to load models');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleModelChange = (index: number, modelPath: string) => {
    const newSelected = [...selectedModels];
    newSelected[index] = modelPath;
    setSelectedModels(newSelected);
  };

  const handleUseSameModel = (modelPath: string) => {
    if (modelPath) {
      setSelectedModels([modelPath, modelPath, modelPath]);
    }
  };

  const handleStart = () => {
    if (selectedModels.every(path => path !== '')) {
      onStart(selectedModels);
    }
  };

  if (loading) {
    return <div>Loading models...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
      <h2>Select AI Models</h2>
      <p>Choose models for 3 AI players (you will be player 0)</p>

      {models.length === 0 && (
        <div style={{ color: 'red', marginBottom: '20px' }}>
          No models available. Please train models first.
        </div>
      )}

      {models.length > 0 && (
        <>
          <div style={{ marginBottom: '20px' }}>
            <label>Use same model for all AI players: </label>
            <select
              onChange={(e) => handleUseSameModel(e.target.value)}
              style={{ padding: '5px', marginLeft: '10px' }}
            >
              <option value="">Select model...</option>
              {models.map((model) => (
                <option key={model.id} value={model.path}>
                  {model.id} {model.elo ? `(ELO: ${model.elo.toFixed(0)})` : ''}
                </option>
              ))}
            </select>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
            {[0, 1, 2].map((index) => (
              <div key={index} style={{ border: '1px solid #ccc', padding: '15px', borderRadius: '5px' }}>
                <label>
                  <strong>AI Player {index + 1}:</strong>
                  <select
                    value={selectedModels[index]}
                    onChange={(e) => handleModelChange(index, e.target.value)}
                    style={{ padding: '5px', marginLeft: '10px', width: '400px' }}
                  >
                    <option value="">Select model...</option>
                    {models.map((model) => (
                      <option key={model.id} value={model.path}>
                        {model.id}
                        {model.step ? ` (step: ${model.step})` : ''}
                        {model.elo ? ` - ELO: ${model.elo.toFixed(0)}` : ''}
                        {model.winrate ? ` - Winrate: ${(model.winrate * 100).toFixed(1)}%` : ''}
                      </option>
                    ))}
                  </select>
                </label>
                {selectedModels[index] && (
                  <div style={{ marginTop: '5px', fontSize: '0.9em', color: '#666' }}>
                    {(() => {
                      const model = models.find(m => m.path === selectedModels[index]);
                      return model ? (
                        <div>
                          <div>Path: {model.path}</div>
                          {model.step && <div>Training Step: {model.step}</div>}
                          {model.elo && <div>ELO Rating: {model.elo.toFixed(0)}</div>}
                          {model.winrate && <div>Winrate: {(model.winrate * 100).toFixed(1)}%</div>}
                        </div>
                      ) : null;
                    })()}
                  </div>
                )}
              </div>
            ))}
          </div>

          <button
            onClick={handleStart}
            disabled={!selectedModels.every(path => path !== '')}
            style={{
              marginTop: '20px',
              padding: '10px 20px',
              fontSize: '16px',
              backgroundColor: selectedModels.every(path => path !== '') ? '#4CAF50' : '#ccc',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: selectedModels.every(path => path !== '') ? 'pointer' : 'not-allowed',
            }}
          >
            Start Game
          </button>
        </>
      )}
    </div>
  );
};

