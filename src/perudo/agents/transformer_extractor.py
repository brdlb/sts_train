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
    Transformer-based features extractor for processing action history sequences.
    
    Architecture:
    - Embeddings for action_type, bid quantities and values
    - Positional encodings
    - Multi-layer transformer encoder for deep sequence processing
    - Attention pooling for aggregating sequence information (focuses on important elements)
    - Post-transformer normalization for training stability
    - Enhanced MLP for static information processing (3 layers with GELU activation)
    - Combined output features from history and static information
    
    Processes action history with action_type (bid/challenge/believe) and encoded_bid.
    Uses attention pooling instead of simple averaging to better focus on relevant sequence elements.
    """
    
    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        embed_dim: int = 128,
        dim_feedforward: int = 512,
        max_history_length: int = 40,
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
        
        # Embeddings for action_type, bid quantities and values
        # Action type: 0=bid, 1=challenge, 2=believe, so 3 values
        # Quantity: 1 to max_quantity (decoded from encoded_bid), so max_quantity + 1 values (0 is padding)
        # Value: 1 to 6 (dice values, decoded from encoded_bid), so 7 values (0-6, where 0 is padding)
        # Distribute embed_dim across three embeddings, ensuring sum equals embed_dim
        embed_dim_per_feature = embed_dim // 3
        # Calculate actual concatenated size (may be slightly less than embed_dim due to integer division)
        concatenated_embed_dim = embed_dim_per_feature * 3
        self.action_type_embedding = nn.Embedding(3, embed_dim_per_feature)  # 0=bid, 1=challenge, 2=believe
        self.quantity_embedding = nn.Embedding(max_quantity + 1, embed_dim_per_feature)  # 0 is padding
        self.value_embedding = nn.Embedding(7, embed_dim_per_feature)  # 0-6, where 0 is padding
        
        # Projection to combine action_type, quantity and value embeddings
        # Use actual concatenated dimension as input, project to embed_dim
        self.bid_projection = nn.Linear(concatenated_embed_dim, embed_dim)
        
        # Layer normalization after embedding (improves training stability)
        self.embed_norm = nn.LayerNorm(embed_dim)
        
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
        
        # Attention pooling for aggregating transformer output
        # Uses learnable query token to attend to all sequence positions
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Learnable query token for attention pooling
        self.aggregation_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Post-transformer normalization
        self.post_transformer_norm = nn.LayerNorm(embed_dim)
        
        # MLP for static information (enhanced for better integration)
        # Get static_info size from observation space
        static_info_size = observation_space['static_info'].shape[0]
        self.static_mlp = nn.Sequential(
            nn.Linear(static_info_size, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Output projection to features_dim
        # Combined: transformer output (embed_dim) + static features (embed_dim)
        combined_dim = embed_dim + embed_dim
        self.output_projection = nn.Linear(combined_dim, features_dim)
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through transformer extractor.
        
        Args:
            observations: Dictionary with 'bid_history' and 'static_info' keys
                - bid_history: (batch_size, max_history_length, 2) - (action_type, encoded_bid)
                - static_info: (batch_size, static_info_size)
        
        Returns:
            Features tensor of shape (batch_size, features_dim)
        """
        bid_history = observations["bid_history"]  # (batch_size, max_history_length, 2)
        static_info = observations["static_info"]  # (batch_size, static_info_size)
        
        batch_size = bid_history.shape[0]
        
        # Extract action_type and encoded_bid
        action_types = bid_history[:, :, 0].long()  # (batch_size, max_history_length)
        encoded_bids = bid_history[:, :, 1].long()  # (batch_size, max_history_length)
        
        # Decode encoded_bid into quantity and value
        # encoded_bid = (quantity - 1) * 6 + (value - 1)
        # quantity = (encoded_bid // 6) + 1
        # value = (encoded_bid % 6) + 1
        quantities = (encoded_bids // 6) + 1  # (batch_size, max_history_length)
        values = (encoded_bids % 6) + 1  # (batch_size, max_history_length)
        
        # For padding positions (encoded_bid == 0), set quantity and value to 0
        padding_positions = (encoded_bids == 0)
        quantities = torch.where(padding_positions, torch.zeros_like(quantities), quantities)
        values = torch.where(padding_positions, torch.zeros_like(values), values)
        
        # Get embeddings
        action_type_embeds = self.action_type_embedding(action_types)  # (batch_size, max_history_length, embed_dim // 3)
        quantity_embeds = self.quantity_embedding(quantities)  # (batch_size, max_history_length, embed_dim // 3)
        value_embeds = self.value_embedding(values)  # (batch_size, max_history_length, embed_dim // 3)
        
        # Combine action_type, quantity and value embeddings
        bid_embeds = torch.cat([action_type_embeds, quantity_embeds, value_embeds], dim=-1)  # (batch_size, max_history_length, embed_dim)
        bid_embeds = self.bid_projection(bid_embeds)  # (batch_size, max_history_length, embed_dim)
        
        # Add positional encodings
        bid_embeds = bid_embeds + self.pos_embedding  # (batch_size, max_history_length, embed_dim)
        
        # Apply layer normalization after embedding (improves training stability)
        bid_embeds = self.embed_norm(bid_embeds)
        
        # Create padding mask (True for padding positions, False for valid positions)
        # Padding is when encoded_bid == 0
        padding_mask = (encoded_bids == 0)  # (batch_size, max_history_length)
        
        # Check if sequences have valid (non-padding) elements
        has_valid_elements = ~padding_mask.all(dim=1)  # (batch_size,) - True if sequence has at least one valid element
        
        # Create modified mask that ensures at least one position is always valid
        # This prevents the transformer from receiving fully masked sequences
        modified_mask = padding_mask.clone()
        all_padded = modified_mask.all(dim=1)  # (batch_size,) - True if sequence is fully padded
        # If all positions are padded, make the first position valid (set to False)
        if all_padded.any():
            modified_mask[all_padded, 0] = False  # Unmask first position for fully padded sequences
        
        # Transformer encoder
        # Note: src_key_padding_mask expects True for positions to ignore (padding)
        transformer_output = self.transformer_encoder(
            bid_embeds,
            src_key_padding_mask=modified_mask
        )  # (batch_size, max_history_length, embed_dim)
        
        # Aggregate transformer output using attention pooling
        # This allows the model to focus on the most important elements in the sequence
        # Expand aggregation query to batch size
        query = self.aggregation_query.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        
        # Attention pooling: query attends to transformer_output
        # key_padding_mask: True for positions to ignore (padding)
        aggregated_history, _ = self.attention_pool(
            query=query,
            key=transformer_output,
            value=transformer_output,
            key_padding_mask=modified_mask,
        )  # (batch_size, 1, embed_dim)
        
        # Remove sequence dimension
        aggregated_history = aggregated_history.squeeze(1)  # (batch_size, embed_dim)
        
        # Apply post-transformer normalization
        aggregated_history = self.post_transformer_norm(aggregated_history)  # (batch_size, embed_dim)
        
        # Process static information
        static_features = self.static_mlp(static_info)  # (batch_size, embed_dim)
        
        # Combine transformer output and static features
        combined_features = torch.cat([aggregated_history, static_features], dim=-1)  # (batch_size, embed_dim + embed_dim)
        
        # Project to final feature dimension
        output_features = self.output_projection(combined_features)  # (batch_size, features_dim)
        
        return output_features

