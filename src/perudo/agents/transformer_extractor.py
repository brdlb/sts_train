"""
Transformer-based features extractor for Perudo RL agent.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Transformer-based features extractor for processing bid history sequences.
    
    Architecture:
    - Embeddings for bid quantities and values
    - Positional encodings
    - Transformer encoder for sequence processing
    - MLP for static information
    - Combined output features
    """
    
    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        embed_dim: int = 128,
        dim_feedforward: int = 512,
        max_history_length: int = 20,
        max_quantity: int = 30,
        dropout: float = 0.1,
    ):
        """
        Initialize transformer features extractor.
        
        Args:
            observation_space: Gymnasium observation space (Dict with 'bid_history' and 'static_info')
            features_dim: Dimension of output features
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
            embed_dim: Dimension of embeddings
            dim_feedforward: Dimension of feedforward network in transformer
            max_history_length: Maximum length of bid history sequence
            max_quantity: Maximum dice quantity in bids
            dropout: Dropout rate
        """
        super().__init__(observation_space, features_dim)
        
        self.embed_dim = embed_dim
        self.max_history_length = max_history_length
        self.max_quantity = max_quantity
        
        # Embeddings for bid quantities and values
        # Quantity: 0 to max_quantity (inclusive), so max_quantity + 1 values
        # Value: 1 to 6 (dice values), so 6 values
        self.quantity_embedding = nn.Embedding(max_quantity + 1, embed_dim // 2)
        self.value_embedding = nn.Embedding(7, embed_dim // 2)  # 0-6, where 0 is padding
        
        # Projection to combine quantity and value embeddings
        self.bid_projection = nn.Linear(embed_dim, embed_dim)
        
        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_history_length, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # MLP for static information
        # Get static_info size from observation space
        static_info_size = observation_space['static_info'].shape[0]
        self.static_mlp = nn.Sequential(
            nn.Linear(static_info_size, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
        )
        
        # Output projection to features_dim
        # Combined: transformer output (embed_dim) + static features (embed_dim // 2)
        combined_dim = embed_dim + embed_dim // 2
        self.output_projection = nn.Linear(combined_dim, features_dim)
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through transformer extractor.
        
        Args:
            observations: Dictionary with 'bid_history' and 'static_info' keys
                - bid_history: (batch_size, max_history_length, 2) - (quantity, value)
                - static_info: (batch_size, static_info_size)
        
        Returns:
            Features tensor of shape (batch_size, features_dim)
        """
        bid_history = observations["bid_history"]  # (batch_size, max_history_length, 2)
        static_info = observations["static_info"]  # (batch_size, static_info_size)
        
        batch_size = bid_history.shape[0]
        
        # Extract quantities and values
        quantities = bid_history[:, :, 0].long()  # (batch_size, max_history_length)
        values = bid_history[:, :, 1].long()  # (batch_size, max_history_length)
        
        # Get embeddings
        quantity_embeds = self.quantity_embedding(quantities)  # (batch_size, max_history_length, embed_dim // 2)
        value_embeds = self.value_embedding(values)  # (batch_size, max_history_length, embed_dim // 2)
        
        # Combine quantity and value embeddings
        bid_embeds = torch.cat([quantity_embeds, value_embeds], dim=-1)  # (batch_size, max_history_length, embed_dim)
        bid_embeds = self.bid_projection(bid_embeds)  # (batch_size, max_history_length, embed_dim)
        
        # Add positional encodings
        bid_embeds = bid_embeds + self.pos_embedding  # (batch_size, max_history_length, embed_dim)
        
        # Create padding mask (True for padding positions, False for valid positions)
        # Padding is when quantity == 0 and value == 0
        padding_mask = (quantities == 0) & (values == 0)  # (batch_size, max_history_length)
        
        # Transformer encoder
        # Note: src_key_padding_mask expects True for positions to ignore (padding)
        transformer_output = self.transformer_encoder(
            bid_embeds,
            src_key_padding_mask=padding_mask
        )  # (batch_size, max_history_length, embed_dim)
        
        # Aggregate transformer output: take the last valid (non-padding) element
        # For simplicity, we'll use attention pooling or take the first non-padding element
        # Here we use a simple approach: take the last position (which should contain context)
        # In practice, we could use learned attention pooling
        aggregated_history = transformer_output[:, -1, :]  # (batch_size, embed_dim)
        
        # Process static information
        static_features = self.static_mlp(static_info)  # (batch_size, embed_dim // 2)
        
        # Combine transformer output and static features
        combined_features = torch.cat([aggregated_history, static_features], dim=-1)  # (batch_size, embed_dim + embed_dim // 2)
        
        # Project to final feature dimension
        output_features = self.output_projection(combined_features)  # (batch_size, features_dim)
        
        return output_features

