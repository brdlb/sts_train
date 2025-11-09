# Perudo RL Environment

Reinforcement Learning environment for training AI agents to play Perudo (Liar's Dice).

## Description

This project implements a Reinforcement Learning (RL) environment for the Perudo game using:
- **PyTorch** - Deep learning framework
- **Stable Baselines3** - RL algorithms library
- **Gymnasium** - Standard for creating game environments
- **TensorBoard** - Training monitoring

## Key Features

- **Parameter Sharing** - Single neural network for all agents, enabling efficient multi-agent training
- **Self-Play with Opponent Pool** - Automatic snapshot management with ELO rating system
- **Vectorized Environment** - Each VecEnv instance represents one table with 3-8 players
- **Global Advantage Normalization** - Advantages normalized across the entire batch
- **Random Player Count** - Each episode randomly selects 3-8 players for training diversity
- **Extended Reward System** - Detailed reward system with intermediate rewards for bluffing, successful challenges, and dice advantage
- **Classic Perudo Rules** - Support for special rounds (Palifico), Paco, and special "ones" rules (Pasari/Jokers)

## Installation

1. Clone the repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install PyTorch with CUDA support (if using GPU):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Basic Training

```bash
python -m src.perudo.training.train
```

### Training with Parameters

```bash
python -m src.perudo.training.train \
    --num-envs 8 \
    --total-timesteps 10000000 \
    --n-steps 1024 \
    --batch-size 128
```

### Command Line Arguments

- `--num-envs` - Number of parallel environments/tables (default: 8)
- `--total-timesteps` - Total training steps (default: 10_000_000)
- `--n-steps` - Steps to collect before update (default: 1024)
- `--batch-size` - Training batch size (default: 128)
- `--device` - Device for training (cpu, cuda, cuda:0). Auto-detects GPU if not specified

**Note**: Ensure that `n_steps × num_envs × num_players` is divisible by `batch_size`.

### View Training Metrics

```bash
tensorboard --logdir=logs/
```

## Architecture

### Parameter Sharing

Single PPO model for all agents, identified via agent_id in observation. This enables:
- Efficient data usage from all agents
- Agent specialization through agent_id
- Reduced memory and computational requirements

### Vectorized Environment

`PerudoMultiAgentVecEnv` creates multiple parallel environments (tables):
- **Agent 0** - Learning agent (uses current PPO model)
- **Agents 1-N** - Opponents selected from opponent pool

Opponents are selected from the pool based on winrate statistics at each `reset()`.

### Opponent Pool

Manages policy snapshots for self-play:
- **Automatic Snapshots** - Saved every N steps (default: 50k, configurable via `snapshot_freq`)
- **Pool Management** - Stores last 10-20 snapshots (max_pool_size=20, min_pool_size=10)
- **Weighted Selection** - Based on winrate, ELO rating, and number of games
- **ELO System** - Each snapshot has ELO rating (initial: 1500, K-factor: 32)
- **Auto Cleanup** - Removes old snapshots while keeping best by ELO (keep_best=3)
- **Metadata** - Statistics stored in `models/opponent_pool/pool_metadata.json`

## Project Structure

```
.
├── src/perudo/
│   ├── game/              # Perudo game logic
│   ├── agents/            # RL agents
│   ├── training/          # Training modules
│   └── utils/             # Helper functions
├── tests/                 # Tests
├── logs/                  # Logs and metrics
├── models/                # Saved models
│   └── opponent_pool/     # Opponent snapshots
└── requirements.txt
```

## Configuration

Training parameters are configured in `src/perudo/training/config.py`. Key parameters:

- **Game**: `num_players`, `dice_per_player`, `max_quantity`
- **Training**: `learning_rate`, `n_steps`, `batch_size`, `gamma`, `gae_lambda`
- **Opponent Pool**: `max_pool_size`, `snapshot_freq`, `elo_k`

## Game Rules (Brief)

Perudo is a dice game with bluffing and incomplete information:
- Each player starts with 5 dice
- Players make bids on total dice count of a specific value
- **Challenge** - Call out previous player's bid
- **Believe** - All players reveal dice, exact match check
- **Special Round (Palifico)** - Activated when player has 1 die (ones are not jokers)
- Last player with dice wins

## Testing

Run all tests:
```bash
pytest tests/ -v
```

## License

MIT License
