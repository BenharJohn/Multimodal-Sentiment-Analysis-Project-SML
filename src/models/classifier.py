"""
Classification Head
MLP-based classifier for sentiment prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron classifier.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list = None,
        dropout: float = 0.3,
        activation: str = "relu",
        use_batch_norm: bool = True
    ):
        """
        Initialize MLP classifier.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions (e.g., [512, 256])
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'tanh')
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [input_dim // 2]

        # Build layers
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation}")

            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, features):
        """
        Forward pass through classifier.

        Args:
            features: [batch_size, input_dim] input features

        Returns:
            logits: [batch_size, num_classes] class logits
        """
        return self.classifier(features)


class AttentionClassifier(nn.Module):
    """
    Classifier with attention mechanism over input features.
    Useful when input is a concatenation of multiple feature streams.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = None,
        dropout: float = 0.3,
        num_attention_heads: int = 4
    ):
        """
        Initialize attention-based classifier.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for attention
            dropout: Dropout probability
            num_attention_heads: Number of attention heads
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        self.input_dim = input_dim
        self.num_classes = num_classes

        # Self-attention over features
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norm
        self.ln = nn.LayerNorm(input_dim)

        # MLP classifier
        self.classifier = MLPClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=[hidden_dim],
            dropout=dropout
        )

    def forward(self, features):
        """
        Forward pass with attention.

        Args:
            features: [batch_size, input_dim] or [batch_size, seq_len, input_dim]

        Returns:
            logits: [batch_size, num_classes]
        """
        # If 2D, expand to sequence
        if features.dim() == 2:
            features = features.unsqueeze(1)  # [B, 1, D]

        # Self-attention
        attn_out, _ = self.attention(features, features, features)
        attn_out = self.ln(attn_out + features)

        # Pool sequence (mean pooling)
        pooled = attn_out.mean(dim=1)  # [B, D]

        # Classify
        logits = self.classifier(pooled)

        return logits


class BilinearClassifier(nn.Module):
    """
    Bilinear classifier for multimodal fusion.
    Captures second-order interactions between modalities.
    """

    def __init__(
        self,
        input_dim1: int,
        input_dim2: int,
        num_classes: int,
        hidden_dim: int = None,
        dropout: float = 0.3
    ):
        """
        Initialize bilinear classifier.

        Args:
            input_dim1: Dimension of first modality
            input_dim2: Dimension of second modality
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for MLP (after bilinear)
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.num_classes = num_classes

        # Bilinear layer
        self.bilinear = nn.Bilinear(input_dim1, input_dim2, hidden_dim or num_classes)

        # If using hidden layer, add MLP
        if hidden_dim is not None:
            self.mlp = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.mlp = None

    def forward(self, features1, features2):
        """
        Forward pass with bilinear interaction.

        Args:
            features1: [batch_size, input_dim1]
            features2: [batch_size, input_dim2]

        Returns:
            logits: [batch_size, num_classes]
        """
        # Bilinear fusion
        fused = self.bilinear(features1, features2)

        # Optional MLP
        if self.mlp is not None:
            logits = self.mlp(fused)
        else:
            logits = fused

        return logits


class EnsembleClassifier(nn.Module):
    """
    Ensemble classifier that combines predictions from multiple heads.
    Useful for multi-task or multi-view learning.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_heads: int = 3,
        hidden_dim: int = None,
        dropout: float = 0.3,
        ensemble_method: str = "average"
    ):
        """
        Initialize ensemble classifier.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of output classes
            num_heads: Number of classifier heads
            hidden_dim: Hidden dimension for each head
            dropout: Dropout probability
            ensemble_method: How to combine predictions ('average', 'vote', 'learned')
        """
        super().__init__()

        self.num_heads = num_heads
        self.num_classes = num_classes
        self.ensemble_method = ensemble_method

        # Create multiple classifier heads
        self.heads = nn.ModuleList([
            MLPClassifier(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dims=[hidden_dim] if hidden_dim else None,
                dropout=dropout
            )
            for _ in range(num_heads)
        ])

        # Learned ensemble weights
        if ensemble_method == "learned":
            self.ensemble_weights = nn.Parameter(torch.ones(num_heads) / num_heads)

    def forward(self, features):
        """
        Forward pass through ensemble.

        Args:
            features: [batch_size, input_dim]

        Returns:
            logits: [batch_size, num_classes] ensembled predictions
            individual_logits: [batch_size, num_heads, num_classes] (optional)
        """
        # Get predictions from all heads
        all_logits = []
        for head in self.heads:
            logits = head(features)
            all_logits.append(logits)

        # Stack: [batch_size, num_heads, num_classes]
        all_logits = torch.stack(all_logits, dim=1)

        # Ensemble
        if self.ensemble_method == "average":
            # Simple average
            final_logits = all_logits.mean(dim=1)

        elif self.ensemble_method == "vote":
            # Majority vote (argmax then mode)
            preds = all_logits.argmax(dim=-1)  # [B, num_heads]
            # Convert to one-hot and sum
            one_hot = F.one_hot(preds, num_classes=self.num_classes).float()
            final_logits = one_hot.sum(dim=1)  # Vote counts

        elif self.ensemble_method == "learned":
            # Learned weighted average
            weights = F.softmax(self.ensemble_weights, dim=0)
            weights = weights.view(1, -1, 1)  # [1, num_heads, 1]
            final_logits = (all_logits * weights).sum(dim=1)

        else:
            raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")

        return final_logits


class SentimentClassifier(nn.Module):
    """
    Specialized sentiment classifier with domain-specific design.
    Includes auxiliary losses for aspect and polarity prediction.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 3,  # Positive, Negative, Neutral
        hidden_dims: list = None,
        dropout: float = 0.3,
        use_ordinal: bool = False
    ):
        """
        Initialize sentiment classifier.

        Args:
            input_dim: Input feature dimension
            num_classes: Number of sentiment classes (default: 3)
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
            use_ordinal: Whether to use ordinal regression for sentiment
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.use_ordinal = use_ordinal

        if hidden_dims is None:
            hidden_dims = [512, 256]

        # Main classifier
        self.classifier = MLPClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_batch_norm=True
        )

        # Ordinal regression (if enabled)
        if use_ordinal:
            # Binary classifiers for ordinal thresholds
            # E.g., for 3 classes: is_positive?, is_neutral_or_positive?
            self.ordinal_heads = nn.ModuleList([
                nn.Linear(hidden_dims[-1], 1)
                for _ in range(num_classes - 1)
            ])

    def forward(self, features):
        """
        Forward pass.

        Args:
            features: [batch_size, input_dim]

        Returns:
            dict with:
                - logits: [batch_size, num_classes] main sentiment logits
                - ordinal_logits: list of [batch_size, 1] ordinal predictions (optional)
        """
        logits = self.classifier(features)

        output = {'logits': logits}

        if self.use_ordinal:
            # Extract intermediate features for ordinal heads
            # (This requires modifying classifier to expose intermediate features)
            # For now, just use the logits
            ordinal_logits = []
            for head in self.ordinal_heads:
                ord_logit = head(features)  # Simplified
                ordinal_logits.append(ord_logit)
            output['ordinal_logits'] = ordinal_logits

        return output
