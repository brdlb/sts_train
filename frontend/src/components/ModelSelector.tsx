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
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-300 text-lg">Loading models...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-red-400 text-lg">Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-gray-800 text-white overflow-y-auto">
      <div className="flex-1 flex flex-col items-center justify-center p-8">
        <div className="max-w-2xl w-full">
          <h2 className="text-4xl font-bold text-center mb-2 text-orange-400">Select AI Models</h2>
          <p className="text-center text-gray-300 mb-8">Choose models for 3 AI players (you will be player 0)</p>

          {models.length === 0 && (
            <div className="bg-gray-700 rounded-lg p-8 text-center">
              <div className="text-6xl mb-4">🎲</div>
              <h3 className="text-2xl font-semibold text-white mb-2">No Models Available</h3>
              <p className="text-gray-300 mb-4">
                You need to train models before you can start a game.
              </p>
              <p className="text-sm text-gray-400">
                Please train models first using the training scripts.
              </p>
            </div>
          )}

          {models.length > 0 && (
            <div className="space-y-6">
              <div className="bg-gray-700 rounded-lg p-4">
                <label className="block text-sm font-semibold text-gray-200 mb-2">
                  Use same model for all AI players:
                </label>
                <select
                  onChange={(e) => handleUseSameModel(e.target.value)}
                  className="w-full px-4 py-2 bg-gray-600 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500 hover:bg-gray-550 transition-colors"
                >
                  <option value="">Select model...</option>
                  {models.map((model) => (
                    <option key={model.id} value={model.path}>
                      {model.id} {model.elo ? `(ELO: ${model.elo.toFixed(0)})` : ''}
                    </option>
                  ))}
                </select>
              </div>

              <div className="grid grid-cols-1 gap-4">
                {[0, 1, 2].map((index) => (
                  <div key={index} className="bg-gray-700 rounded-lg p-4">
                    <label className="block text-sm font-semibold text-gray-200 mb-3">
                      AI Player {index + 1}
                    </label>
                    <select
                      value={selectedModels[index]}
                      onChange={(e) => handleModelChange(index, e.target.value)}
                      className="w-full px-4 py-2 bg-gray-600 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500 hover:bg-gray-550 transition-colors mb-3"
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
                    {selectedModels[index] && (
                      <div className="bg-gray-600 rounded p-3 text-sm text-gray-200">
                        {(() => {
                          const model = models.find(m => m.path === selectedModels[index]);
                          return model ? (
                            <div className="space-y-1">
                              <div><span className="text-gray-400">Path:</span> {model.path}</div>
                              {model.step && <div><span className="text-gray-400">Training Step:</span> {model.step}</div>}
                              {model.elo && <div><span className="text-gray-400">ELO Rating:</span> {model.elo.toFixed(0)}</div>}
                              {model.winrate && <div><span className="text-gray-400">Winrate:</span> {(model.winrate * 100).toFixed(1)}%</div>}
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
                className="w-full px-6 py-3 rounded-lg font-semibold text-white transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:bg-gray-600 enabled:bg-green-600 enabled:hover:bg-green-700"
              >
                Start Game
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

