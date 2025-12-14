"""
Multi-Encoder Fusion Module
Attention-based fusion of CLIP and auxiliary encoder features (BERT/ResNet).
Implements the fusion mechanism from CDAN 2025 paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class AttentionEncoderFusion(nn.Module):
    """
    Attention-based fusion of two encoder outputs.

    Uses cross-attention to combine CLIP features with auxiliary encoder features
    (BERT for text, ResNet for images).

    Architecture:
        CLIP features ─┬─→ Cross-Attention ─→ Fused features
        Aux features ──┘

    The CLIP features serve as the query (what to focus on),
    while auxiliary features serve as key/value (additional information).
    """

    def __init__(
        self,
        clip_dim: int,
        aux_dim: int,
        output_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        """
        Initialize attention-based encoder fusion.

        Args:
            clip_dim: Dimension of CLIP features (512 for CLIP base)
            aux_dim: Dimension of auxiliary features (768 for BERT, 2048 for ResNet50)
            output_dim: Output dimension after fusion
            num_heads: Number of attention heads
            dropout: Dropout probability
            use_residual: Whether to use residual connection from CLIP features
        """
        super().__init__()

        self.clip_dim = clip_dim
        self.aux_dim = aux_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.use_residual = use_residual

        # Project both to same dimension for attention
        self.hidden_dim = output_dim
        self.head_dim = self.hidden_dim // num_heads
        assert self.hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Projections for attention
        self.clip_proj = nn.Linear(clip_dim, self.hidden_dim)
        self.aux_proj = nn.Linear(aux_dim, self.hidden_dim)

        # Query from CLIP, Key/Value from auxiliary
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(self.hidden_dim, output_dim)

        # Layer norm and dropout
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Feed-forward network for additional processing
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim),
            nn.Dropout(dropout)
        )

        # Residual projection if dimensions differ
        if clip_dim != output_dim:
            self.residual_proj = nn.Linear(clip_dim, output_dim)
        else:
            self.residual_proj = nn.Identity()

    def forward(
        self,
        clip_features: torch.Tensor,
        aux_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse CLIP and auxiliary features using attention.

        Args:
            clip_features: [batch_size, clip_dim] CLIP encoder output
            aux_features: [batch_size, aux_dim] Auxiliary encoder output (BERT/ResNet)

        Returns:
            fused: [batch_size, output_dim] Fused features
        """
        batch_size = clip_features.size(0)

        # Project to common dimension
        clip_hidden = self.clip_proj(clip_features)  # [B, hidden_dim]
        aux_hidden = self.aux_proj(aux_features)     # [B, hidden_dim]

        # Normalize
        clip_hidden = self.norm1(clip_hidden)
        aux_hidden = self.norm1(aux_hidden)

        # Prepare for multi-head attention
        # Add sequence dimension: [B, hidden_dim] -> [B, 1, hidden_dim]
        clip_seq = clip_hidden.unsqueeze(1)  # [B, 1, hidden_dim]
        aux_seq = aux_hidden.unsqueeze(1)    # [B, 1, hidden_dim]

        # Compute Q, K, V
        Q = self.q_proj(clip_seq)  # [B, 1, hidden_dim]
        K = self.k_proj(aux_seq)   # [B, 1, hidden_dim]
        V = self.v_proj(aux_seq)   # [B, 1, hidden_dim]

        # Reshape for multi-head attention: [B, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [B, heads, 1, 1]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, heads, 1, head_dim]

        # Reshape back: [B, 1, hidden_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, 1, self.hidden_dim)
        attn_output = attn_output.squeeze(1)  # [B, hidden_dim]

        # Output projection
        fused = self.out_proj(attn_output)  # [B, output_dim]
        fused = self.dropout(fused)

        # Residual connection from CLIP features
        if self.use_residual:
            residual = self.residual_proj(clip_features)
            fused = fused + residual

        # FFN with residual
        fused = fused + self.ffn(self.norm2(fused))

        return fused


class GatedEncoderFusion(nn.Module):
    """
    Gated fusion of two encoder outputs.

    Alternative fusion method using learned gates to weight contributions
    from CLIP and auxiliary encoders.
    """

    def __init__(
        self,
        clip_dim: int,
        aux_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize gated encoder fusion.

        Args:
            clip_dim: Dimension of CLIP features
            aux_dim: Dimension of auxiliary features
            output_dim: Output dimension after fusion
            dropout: Dropout probability
        """
        super().__init__()

        self.clip_dim = clip_dim
        self.aux_dim = aux_dim
        self.output_dim = output_dim

        # Project to common dimension
        self.clip_proj = nn.Linear(clip_dim, output_dim)
        self.aux_proj = nn.Linear(aux_dim, output_dim)

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(clip_dim + aux_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, 2),
            nn.Softmax(dim=-1)
        )

        # Output normalization
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        clip_features: torch.Tensor,
        aux_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse CLIP and auxiliary features using gating.

        Args:
            clip_features: [batch_size, clip_dim]
            aux_features: [batch_size, aux_dim]

        Returns:
            fused: [batch_size, output_dim]
        """
        # Project to common dimension
        clip_proj = self.clip_proj(clip_features)  # [B, output_dim]
        aux_proj = self.aux_proj(aux_features)     # [B, output_dim]

        # Compute gate weights
        concat = torch.cat([clip_features, aux_features], dim=-1)  # [B, clip_dim + aux_dim]
        gate_weights = self.gate(concat)  # [B, 2]

        # Apply gates
        fused = gate_weights[:, 0:1] * clip_proj + gate_weights[:, 1:2] * aux_proj

        # Normalize and dropout
        fused = self.norm(fused)
        fused = self.dropout(fused)

        return fused


class SelfAdaptiveAggregation(nn.Module):
    """
    Self-Adaptive Weighted Aggregation module from CDAN 2025 paper.

    Combines multiple feature streams with learned adaptive weights.
    """

    def __init__(
        self,
        input_dims: list,
        output_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize self-adaptive aggregation.

        Args:
            input_dims: List of input dimensions for each feature stream
            output_dim: Output dimension after aggregation
            dropout: Dropout probability
        """
        super().__init__()

        self.num_streams = len(input_dims)
        self.output_dim = output_dim

        # Project each stream to common dimension
        self.projections = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])

        # Adaptive weight computation
        self.weight_net = nn.Sequential(
            nn.Linear(sum(input_dims), 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, self.num_streams),
            nn.Softmax(dim=-1)
        )

        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate multiple feature streams.

        Args:
            *features: Variable number of feature tensors

        Returns:
            aggregated: [batch_size, output_dim] aggregated features
            weights: [batch_size, num_streams] adaptive weights
        """
        assert len(features) == self.num_streams

        # Project all features to common dimension
        projected = [proj(feat) for proj, feat in zip(self.projections, features)]

        # Compute adaptive weights
        concat = torch.cat(features, dim=-1)
        weights = self.weight_net(concat)  # [B, num_streams]

        # Weighted aggregation
        aggregated = sum(w.unsqueeze(-1) * p for w, p in zip(weights.unbind(dim=-1), projected))

        # Normalize
        aggregated = self.norm(aggregated)
        aggregated = self.dropout(aggregated)

        return aggregated, weights
