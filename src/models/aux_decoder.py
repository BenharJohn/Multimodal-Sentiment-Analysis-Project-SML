"""
Auxiliary Reconstruction Decoder
Reconstructs frozen CLIP pooled embeddings from fused representations.
Acts as a regularizer to preserve original semantic information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxiliaryDecoder(nn.Module):
    """
    Auxiliary decoder that reconstructs original CLIP embeddings.
    This acts as a regularization loss to ensure fused features retain
    semantic information from the original encoders.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        loss_type: str = "cosine"
    ):
        """
        Initialize auxiliary decoder.

        Args:
            input_dim: Dimension of fused features
            output_dim: Dimension of target CLIP embeddings
            hidden_dim: Hidden layer dimension (defaults to mean of input/output)
            num_layers: Number of MLP layers
            dropout: Dropout probability
            loss_type: Loss function ('cosine', 'l2', 'l1', 'smooth_l1')
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.loss_type = loss_type

        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2

        # Build MLP decoder
        layers = []
        current_dim = input_dim

        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer: project to output dimension
                layers.append(nn.Linear(current_dim, output_dim))
            else:
                # Hidden layers
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                current_dim = hidden_dim

        self.decoder = nn.Sequential(*layers)

        # Optional L2 normalization for cosine similarity
        self.normalize = (loss_type == "cosine")

    def forward(self, fused_features, target_embeddings=None):
        """
        Reconstruct CLIP embeddings from fused features.

        Args:
            fused_features: [batch_size, input_dim] fused multimodal features
            target_embeddings: [batch_size, output_dim] target CLIP embeddings (optional)

        Returns:
            dict with:
                - reconstructed: [batch_size, output_dim] reconstructed embeddings
                - loss: scalar reconstruction loss (if target provided)
        """
        # Decode
        reconstructed = self.decoder(fused_features)

        # Normalize if using cosine loss
        if self.normalize:
            reconstructed = F.normalize(reconstructed, p=2, dim=-1)

        output = {'reconstructed': reconstructed}

        # Compute loss if target is provided
        if target_embeddings is not None:
            # Ensure target is normalized for cosine
            if self.normalize:
                target_embeddings = F.normalize(target_embeddings, p=2, dim=-1)

            loss = self.compute_loss(reconstructed, target_embeddings)
            output['loss'] = loss

        return output

    def compute_loss(self, pred, target):
        """
        Compute reconstruction loss.

        Args:
            pred: [batch_size, output_dim] predicted embeddings
            target: [batch_size, output_dim] target embeddings

        Returns:
            scalar loss
        """
        if self.loss_type == "cosine":
            # Cosine similarity loss (1 - cosine similarity)
            # Range: [0, 2], where 0 = identical, 2 = opposite
            cos_sim = F.cosine_similarity(pred, target, dim=-1)
            loss = (1 - cos_sim).mean()

        elif self.loss_type == "l2":
            # L2 (MSE) loss
            loss = F.mse_loss(pred, target)

        elif self.loss_type == "l1":
            # L1 (MAE) loss
            loss = F.l1_loss(pred, target)

        elif self.loss_type == "smooth_l1":
            # Smooth L1 (Huber) loss
            loss = F.smooth_l1_loss(pred, target)

        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        return loss


class DualAuxiliaryDecoder(nn.Module):
    """
    Dual auxiliary decoder that reconstructs both text and image CLIP embeddings.
    Useful when fusing cross-attended text and image features.
    """

    def __init__(
        self,
        input_dim: int,
        text_output_dim: int,
        image_output_dim: int,
        hidden_dim: int = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        loss_type: str = "cosine"
    ):
        """
        Initialize dual auxiliary decoder.

        Args:
            input_dim: Dimension of fused features
            text_output_dim: Dimension of text CLIP embeddings
            image_output_dim: Dimension of image CLIP embeddings
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            dropout: Dropout probability
            loss_type: Loss function type
        """
        super().__init__()

        # Text decoder
        self.text_decoder = AuxiliaryDecoder(
            input_dim=input_dim,
            output_dim=text_output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            loss_type=loss_type
        )

        # Image decoder
        self.image_decoder = AuxiliaryDecoder(
            input_dim=input_dim,
            output_dim=image_output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            loss_type=loss_type
        )

    def forward(
        self,
        text_fused,
        image_fused,
        text_target=None,
        image_target=None
    ):
        """
        Reconstruct both text and image embeddings.

        Args:
            text_fused: [batch_size, input_dim] fused text features
            image_fused: [batch_size, input_dim] fused image features
            text_target: [batch_size, text_output_dim] target text embeddings
            image_target: [batch_size, image_output_dim] target image embeddings

        Returns:
            dict with:
                - text_reconstructed: [batch_size, text_output_dim]
                - image_reconstructed: [batch_size, image_output_dim]
                - text_loss: scalar (if target provided)
                - image_loss: scalar (if target provided)
                - total_loss: scalar (if both targets provided)
        """
        # Decode text
        text_output = self.text_decoder(text_fused, text_target)

        # Decode image
        image_output = self.image_decoder(image_fused, image_target)

        output = {
            'text_reconstructed': text_output['reconstructed'],
            'image_reconstructed': image_output['reconstructed']
        }

        # Combine losses
        if 'loss' in text_output:
            output['text_loss'] = text_output['loss']
        if 'loss' in image_output:
            output['image_loss'] = image_output['loss']

        if 'text_loss' in output and 'image_loss' in output:
            output['total_loss'] = output['text_loss'] + output['image_loss']

        return output


class ContrastiveAuxDecoder(nn.Module):
    """
    Auxiliary decoder with contrastive learning objective.
    Maximizes similarity between reconstructed and target embeddings,
    while minimizing similarity with negative samples.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 0.07
    ):
        """
        Initialize contrastive auxiliary decoder.

        Args:
            input_dim: Dimension of fused features
            output_dim: Dimension of target embeddings
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            dropout: Dropout probability
            temperature: Temperature for contrastive loss
        """
        super().__init__()

        self.decoder = AuxiliaryDecoder(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            loss_type="cosine"  # Use cosine for contrastive
        )

        self.temperature = temperature

    def forward(self, fused_features, target_embeddings):
        """
        Forward pass with contrastive loss.

        Args:
            fused_features: [batch_size, input_dim]
            target_embeddings: [batch_size, output_dim]

        Returns:
            dict with reconstructed embeddings and contrastive loss
        """
        # Decode
        output = self.decoder(fused_features)
        reconstructed = output['reconstructed']

        # Normalize
        reconstructed = F.normalize(reconstructed, p=2, dim=-1)
        target_embeddings = F.normalize(target_embeddings, p=2, dim=-1)

        # Compute similarity matrix
        # Shape: [batch_size, batch_size]
        logits = torch.matmul(reconstructed, target_embeddings.t()) / self.temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(logits.size(0), device=logits.device)

        # Contrastive loss (cross-entropy)
        loss = F.cross_entropy(logits, labels)

        output['loss'] = loss
        output['contrastive_logits'] = logits

        return output
