"""
Cross-Attention Blocks for Multimodal Fusion
Implements bidirectional cross-attention between text and image modalities.
Inspired by CDAN (Cross-Domain Attention Network) style fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention module."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Initialize multi-head cross-attention.

        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in projections
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, return_attention=False):
        """
        Forward pass for cross-attention.

        Args:
            query: [batch_size, query_len, embed_dim]
            key: [batch_size, key_len, embed_dim]
            value: [batch_size, value_len, embed_dim] (same as key)
            key_padding_mask: [batch_size, key_len] mask (True = ignore)
            return_attention: Whether to return attention weights

        Returns:
            output: [batch_size, query_len, embed_dim]
            attention_weights: [batch_size, num_heads, query_len, key_len] (optional)
        """
        batch_size, query_len, embed_dim = query.size()
        _, key_len, _ = key.size()

        # Project and reshape to [batch_size, num_heads, seq_len, head_dim]
        Q = self.q_proj(query).view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores: [batch_size, num_heads, query_len, key_len]
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # Expand mask to [batch_size, 1, 1, key_len]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))

        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values: [batch_size, num_heads, query_len, head_dim]
        output = torch.matmul(attn_weights, V)

        # Reshape back to [batch_size, query_len, embed_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, query_len, embed_dim)

        # Final projection
        output = self.out_proj(output)

        if return_attention:
            return output, attn_weights
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(
        self,
        embed_dim: int,
        ff_dim: int = None,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Initialize feed-forward network.

        Args:
            embed_dim: Input/output dimension
            ff_dim: Hidden dimension (defaults to 4 * embed_dim)
            dropout: Dropout probability
            activation: Activation function name
        """
        super().__init__()
        ff_dim = ff_dim or 4 * embed_dim

        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        """Forward pass."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention block with residual connections and layer normalization.
    Query comes from one modality, Key/Value from another.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: int = None,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        """
        Initialize cross-attention block.

        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout probability
            activation: Activation function for FFN
        """
        super().__init__()

        # Cross-attention
        self.cross_attn = MultiHeadCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Feed-forward
        self.ffn = FeedForward(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            activation=activation
        )

        # Layer normalization
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, return_attention=False):
        """
        Forward pass through cross-attention block.

        Args:
            query: [batch_size, query_len, embed_dim] - target modality
            key: [batch_size, key_len, embed_dim] - source modality
            value: [batch_size, value_len, embed_dim] - source modality
            key_padding_mask: [batch_size, key_len] mask
            return_attention: Whether to return attention weights

        Returns:
            output: [batch_size, query_len, embed_dim]
            attention_weights: Optional attention weights
        """
        # Cross-attention with residual
        attn_output = self.cross_attn(
            query=self.ln1(query),
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            return_attention=return_attention
        )

        if return_attention:
            attn_output, attn_weights = attn_output

        query = query + self.dropout(attn_output)

        # Feed-forward with residual
        ffn_output = self.ffn(self.ln2(query))
        output = query + ffn_output

        if return_attention:
            return output, attn_weights
        return output


class BidirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention between two modalities.
    Text attends to Image, and Image attends to Text simultaneously.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        pool_method: str = "mean"
    ):
        """
        Initialize bidirectional cross-attention.

        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            num_layers: Number of cross-attention layers
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout probability
            activation: Activation function
            pool_method: Pooling method for sequence outputs ('mean', 'max', 'cls')
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.pool_method = pool_method

        # Text -> Image cross-attention layers
        self.text_to_image_layers = nn.ModuleList([
            CrossAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_layers)
        ])

        # Image -> Text cross-attention layers
        self.image_to_text_layers = nn.ModuleList([
            CrossAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                activation=activation
            )
            for _ in range(num_layers)
        ])

    def pool_sequence(self, sequence, method='mean', mask=None):
        """
        Pool sequence to single vector.

        Args:
            sequence: [batch_size, seq_len, embed_dim]
            method: Pooling method ('mean', 'max', 'cls')
            mask: [batch_size, seq_len] attention mask (1 = valid, 0 = padding)

        Returns:
            pooled: [batch_size, embed_dim]
        """
        if method == 'cls':
            # Take first token (CLS token)
            return sequence[:, 0, :]
        elif method == 'mean':
            if mask is not None:
                # Masked mean
                mask = mask.unsqueeze(-1).float()  # [B, T, 1]
                masked_sum = (sequence * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                return masked_sum / counts
            else:
                return sequence.mean(dim=1)
        elif method == 'max':
            if mask is not None:
                # Masked max
                mask = mask.unsqueeze(-1).float()
                sequence = sequence.masked_fill(mask == 0, float('-inf'))
            return sequence.max(dim=1)[0]
        else:
            raise ValueError(f"Unsupported pooling method: {method}")

    def forward(
        self,
        text_tokens,
        vision_patches,
        text_mask=None,
        return_attention=False
    ):
        """
        Bidirectional cross-attention forward pass.

        Args:
            text_tokens: [batch_size, text_len, embed_dim]
            vision_patches: [batch_size, patch_len, embed_dim]
            text_mask: [batch_size, text_len] attention mask (1 = valid, 0 = padding)
            return_attention: Whether to return attention maps

        Returns:
            dict with:
                - text_fused: [batch_size, embed_dim] pooled text after fusion
                - image_fused: [batch_size, embed_dim] pooled image after fusion
                - text_tokens_fused: [batch_size, text_len, embed_dim] (optional)
                - image_tokens_fused: [batch_size, patch_len, embed_dim] (optional)
                - attention_weights: dict of attention weights (optional)
        """
        # Convert mask: 1 = valid, 0 = padding -> False = valid, True = padding
        key_padding_mask = None
        if text_mask is not None:
            key_padding_mask = (text_mask == 0)  # Invert: True = padding

        attention_maps = {} if return_attention else None

        # Apply cross-attention layers
        text_out = text_tokens
        image_out = vision_patches

        for i in range(self.num_layers):
            # Text queries Image (text attends to image)
            text_out_new = self.text_to_image_layers[i](
                query=text_out,
                key=image_out,
                value=image_out,
                key_padding_mask=None,  # No mask for image patches
                return_attention=return_attention
            )

            if return_attention:
                text_out_new, t2i_attn = text_out_new
                attention_maps[f'text_to_image_layer_{i}'] = t2i_attn

            # Image queries Text (image attends to text)
            image_out_new = self.image_to_text_layers[i](
                query=image_out,
                key=text_out,
                value=text_out,
                key_padding_mask=key_padding_mask,
                return_attention=return_attention
            )

            if return_attention:
                image_out_new, i2t_attn = image_out_new
                attention_maps[f'image_to_text_layer_{i}'] = i2t_attn

            text_out = text_out_new
            image_out = image_out_new

        # Pool to get fixed-size representations
        text_fused = self.pool_sequence(text_out, method=self.pool_method, mask=text_mask)
        image_fused = self.pool_sequence(image_out, method='cls')  # Use CLS token for images

        output = {
            'text_fused': text_fused,           # [B, D]
            'image_fused': image_fused,         # [B, D]
            'text_tokens_fused': text_out,      # [B, T, D]
            'image_tokens_fused': image_out,    # [B, P, D]
        }

        if return_attention:
            output['attention_weights'] = attention_maps

        return output
