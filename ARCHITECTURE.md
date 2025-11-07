# Model Architecture

The model implemented in this project is a **PPO + Transformer Layer** architecture.

## Overview

The architecture consists of two main components:

1.  **Transformer-based Features Extractor**: This component processes the sequence of bids made during the game. It uses a Transformer encoder to capture the context and relationships between the bids in the history.
2.  **Proximal Policy Optimization (PPO)**: The features extracted by the Transformer are then fed into a PPO model from the `stable-baselines3` library. The PPO model is responsible for learning the policy and value functions to play the game.

## Details

-   The `TransformerFeaturesExtractor` class (defined in `src/perudo/agents/transformer_extractor.py`) implements the Transformer-based feature extractor.
-   The `train.py` script (in `src/perudo/training/`) instantiates the PPO model and configures it to use the `TransformerFeaturesExtractor` as its feature extractor.

This approach is **not** a "Transformer instead of PPO" (like a Decision Transformer) because the core of the reinforcement learning algorithm is still PPO. It is also not a "Hybrid PPO + Transformer" in the sense of two separate models working in parallel. Instead, the Transformer is a component *within* the PPO model's policy network.
