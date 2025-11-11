"""
Modality Gating Mechanism
Learns to weight and combine different modalities adaptively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityGate(nn.Module):
    """
    Simple gating mechanism for modality fusion.
    Learns scalar weights for each modality based on their pooled representations.
    """

    def __init__(
        self,
        embed_dim: int,
        num_modalities: int = 2,
        gate_activation: str = "sigmoid",
        dropout: float = 0.1
    ):
        """
        Initialize modality gate.

        Args:
            embed_dim: Dimension of input embeddings
            num_modalities: Number of modalities to gate (default: 2 for text + image)
            gate_activation: Activation for gate ('sigmoid', 'softmax', 'tanh')
            dropout: Dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_modalities = num_modalities
        self.gate_activation = gate_activation

        # Gate network: projects concatenated modalities to gate weights
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_modalities)
        )

    def forward(self, *modality_features):
        """
        Compute gate weights and apply to modality features.

        Args:
            *modality_features: Variable number of [batch_size, embed_dim] tensors

        Returns:
            dict with:
                - gated_features: [batch_size, embed_dim * num_modalities] gated and concatenated
                - gate_weights: [batch_size, num_modalities] gate weights
                - weighted_features: list of [batch_size, embed_dim] weighted modalities
        """
        assert len(modality_features) == self.num_modalities, \
            f"Expected {self.num_modalities} modalities, got {len(modality_features)}"

        batch_size = modality_features[0].size(0)

        # Concatenate all modality features
        # Shape: [batch_size, embed_dim * num_modalities]
        concat_features = torch.cat(modality_features, dim=-1)

        # Compute gate weights
        # Shape: [batch_size, num_modalities]
        gate_logits = self.gate_net(concat_features)

        # Apply activation
        if self.gate_activation == "sigmoid":
            gate_weights = torch.sigmoid(gate_logits)
        elif self.gate_activation == "softmax":
            gate_weights = F.softmax(gate_logits, dim=-1)
        elif self.gate_activation == "tanh":
            gate_weights = torch.tanh(gate_logits)
        else:
            raise ValueError(f"Unsupported gate activation: {self.gate_activation}")

        # Apply gates to each modality
        weighted_features = []
        for i, feat in enumerate(modality_features):
            # Shape: [batch_size, 1] * [batch_size, embed_dim] = [batch_size, embed_dim]
            weighted = gate_weights[:, i:i+1] * feat
            weighted_features.append(weighted)

        # Concatenate weighted features
        gated_features = torch.cat(weighted_features, dim=-1)

        return {
            'gated_features': gated_features,           # [B, D * num_modalities]
            'gate_weights': gate_weights,               # [B, num_modalities]
            'weighted_features': weighted_features      # List of [B, D]
        }


class AttentionGate(nn.Module):
    """
    Attention-based gating mechanism.
    Uses attention to compute importance weights for each modality.
    """

    def __init__(
        self,
        embed_dim: int,
        num_modalities: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize attention gate.

        Args:
            embed_dim: Dimension of input embeddings
            num_modalities: Number of modalities
            dropout: Dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_modalities = num_modalities

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, *modality_features):
        """
        Compute attention-based gate weights.

        Args:
            *modality_features: Variable number of [batch_size, embed_dim] tensors

        Returns:
            dict with:
                - gated_features: [batch_size, embed_dim * num_modalities]
                - gate_weights: [batch_size, num_modalities]
                - weighted_features: list of [batch_size, embed_dim]
        """
        assert len(modality_features) == self.num_modalities

        batch_size = modality_features[0].size(0)

        # Stack modalities: [batch_size, num_modalities, embed_dim]
        stacked = torch.stack(modality_features, dim=1)

        # Compute attention scores for each modality
        # Shape: [batch_size, num_modalities, 1]
        attention_scores = self.attention(stacked)

        # Normalize to get weights: [batch_size, num_modalities]
        gate_weights = F.softmax(attention_scores.squeeze(-1), dim=-1)

        # Apply weights
        weighted_features = []
        for i, feat in enumerate(modality_features):
            weighted = gate_weights[:, i:i+1] * feat
            weighted_features.append(weighted)

        # Concatenate
        gated_features = torch.cat(weighted_features, dim=-1)

        return {
            'gated_features': gated_features,
            'gate_weights': gate_weights,
            'weighted_features': weighted_features
        }


class HierarchicalGate(nn.Module):
    """
    Hierarchical gating that first gates within each modality stream,
    then gates across modalities.
    """

    def __init__(
        self,
        embed_dim: int,
        num_modalities: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize hierarchical gate.

        Args:
            embed_dim: Dimension of embeddings
            num_modalities: Number of modalities
            dropout: Dropout probability
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_modalities = num_modalities

        # Intra-modality gates (one per modality)
        self.intra_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid()
            )
            for _ in range(num_modalities)
        ])

        # Inter-modality gate
        self.inter_gate = ModalityGate(
            embed_dim=embed_dim,
            num_modalities=num_modalities,
            gate_activation="softmax",
            dropout=dropout
        )

    def forward(self, *modality_features):
        """
        Apply hierarchical gating.

        Args:
            *modality_features: Variable number of [batch_size, embed_dim] tensors

        Returns:
            dict with gated features and gate weights
        """
        # Apply intra-modality gates
        intra_gated = []
        intra_weights = []
        for i, feat in enumerate(modality_features):
            gate = self.intra_gates[i](feat)
            gated = gate * feat
            intra_gated.append(gated)
            intra_weights.append(gate)

        # Apply inter-modality gate
        output = self.inter_gate(*intra_gated)
        output['intra_gate_weights'] = intra_weights

        return output
