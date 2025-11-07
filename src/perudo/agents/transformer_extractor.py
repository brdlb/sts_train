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
    - Embeddings for player_id, bid quantities and values
    - Positional encodings
    - Transformer encoder for sequence processing
    - MLP for static information
    - Combined output features
    
    Preserves turn order context by including player_id in bid history.
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
        
        # Get max_players from observation space to determine player_id embedding size
        # static_info contains: agent_id (max_players) + current_bid (2) + dice_count (max_players) + 
        #                       current_player (1) + palifico (max_players) + believe (1) + player_dice (5)
        # Total = max_players + 2 + max_players + 1 + max_players + 1 + 5 = 3 * max_players + 9
        # So: max_players = (static_info_size - 9) / 3
        max_players = 8  # Default fallback
        if hasattr(observation_space, 'spaces') and 'static_info' in observation_space.spaces:
            static_info_size = observation_space.spaces['static_info'].shape[0]
            # Calculate max_players: (static_info_size - 9) / 3
            calculated_max_players = (static_info_size - 9) // 3
            if calculated_max_players > 0 and calculated_max_players <= 10:  # Reasonable range
                max_players = calculated_max_players
        elif hasattr(observation_space, 'max_players'):
            max_players = observation_space.max_players
        
        # Embeddings for player_id, bid quantities and values
        # Player ID: 0 to max_players-1, so max_players values (0 is also valid, not just padding)
        # Quantity: 0 to max_quantity (inclusive), so max_quantity + 1 values
        # Value: 1 to 6 (dice values), so 7 values (0-6, where 0 is padding)
        self.player_id_embedding = nn.Embedding(max_players, embed_dim // 3)
        self.quantity_embedding = nn.Embedding(max_quantity + 1, embed_dim // 3)
        self.value_embedding = nn.Embedding(7, embed_dim // 3)  # 0-6, where 0 is padding
        
        # Projection to combine player_id, quantity and value embeddings
        self.bid_projection = nn.Linear(embed_dim, embed_dim)
        
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
        
        # MLP for static information (simplified for efficiency)
        # Get static_info size from observation space
        static_info_size = observation_space['static_info'].shape[0]
        self.static_mlp = nn.Sequential(
            nn.Linear(static_info_size, embed_dim // 2),  # Direct projection to embed_dim // 2
            nn.ReLU(),
            nn.LayerNorm(embed_dim // 2),  # Layer normalization for stability
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
                - bid_history: (batch_size, max_history_length, 3) - (player_id, quantity, value)
                - static_info: (batch_size, static_info_size)
        
        Returns:
            Features tensor of shape (batch_size, features_dim)
        """
        bid_history = observations["bid_history"]  # (batch_size, max_history_length, 3)
        static_info = observations["static_info"]  # (batch_size, static_info_size)
        
        batch_size = bid_history.shape[0]
        
        # Extract player_id, quantities and values
        player_ids = bid_history[:, :, 0].long()  # (batch_size, max_history_length)
        quantities = bid_history[:, :, 1].long()  # (batch_size, max_history_length)
        values = bid_history[:, :, 2].long()  # (batch_size, max_history_length)
        
        # Get embeddings
        player_id_embeds = self.player_id_embedding(player_ids)  # (batch_size, max_history_length, embed_dim // 3)
        quantity_embeds = self.quantity_embedding(quantities)  # (batch_size, max_history_length, embed_dim // 3)
        value_embeds = self.value_embedding(values)  # (batch_size, max_history_length, embed_dim // 3)
        
        # Combine player_id, quantity and value embeddings
        bid_embeds = torch.cat([player_id_embeds, quantity_embeds, value_embeds], dim=-1)  # (batch_size, max_history_length, embed_dim)
        bid_embeds = self.bid_projection(bid_embeds)  # (batch_size, max_history_length, embed_dim)
        
        # Add positional encodings
        bid_embeds = bid_embeds + self.pos_embedding  # (batch_size, max_history_length, embed_dim)
        
        # Apply layer normalization after embedding (improves training stability)
        bid_embeds = self.embed_norm(bid_embeds)
        
        # Create padding mask (True for padding positions, False for valid positions)
        # Padding is when quantity == 0 and value == 0 (valid bids always have quantity > 0 and value >= 1)
        # Note: player_id == 0 is a valid player, so we only check quantity and value for padding
        padding_mask = (quantities == 0) & (values == 0)  # (batch_size, max_history_length)
        
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
        
        # Aggregate transformer output: weighted average of valid elements (vectorized)
        # This is more efficient than taking only the last element and preserves more information
        # Convert mask: True for padding (to ignore), False for valid positions
        valid_mask = ~modified_mask  # (batch_size, max_history_length) - True for valid positions
        # Expand mask to match transformer_output shape
        valid_mask_expanded = valid_mask.unsqueeze(-1)  # (batch_size, max_history_length, 1)
        # Mask out padding positions by setting them to zero
        masked_output = transformer_output * valid_mask_expanded  # (batch_size, max_history_length, embed_dim)
        # Sum over sequence dimension (only valid positions contribute)
        summed_output = masked_output.sum(dim=1)  # (batch_size, embed_dim)
        # Count valid positions per sequence
        valid_count = valid_mask.sum(dim=1, keepdim=True)  # (batch_size, 1)
        # Avoid division by zero (should not happen due to modified_mask, but safety check)
        valid_count = torch.clamp(valid_count, min=1.0)
        # Compute mean (weighted average)
        aggregated_history = summed_output / valid_count  # (batch_size, embed_dim)
        
        # Process static information
        static_features = self.static_mlp(static_info)  # (batch_size, embed_dim // 2)
        
        # Combine transformer output and static features
        combined_features = torch.cat([aggregated_history, static_features], dim=-1)  # (batch_size, embed_dim + embed_dim // 2)
        
        # Project to final feature dimension
        output_features = self.output_projection(combined_features)  # (batch_size, features_dim)
        
        return output_features

