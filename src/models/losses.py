"""
Custom Loss Functions for Multimodal Sentiment Analysis
Includes Focal Loss for class imbalance and Supervised Contrastive Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    From: "Focal Loss for Dense Object Detection" (Lin et al., 2017)

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where:
        - p_t is the probability of the correct class
        - alpha_t is the class weight for the target class
        - gamma is the focusing parameter (reduces loss for well-classified examples)
    """

    def __init__(
        self,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Initialize Focal Loss.

        Args:
            alpha: Class weights. If None, no class weighting is applied.
                   Should be a list of weights for each class [w0, w1, w2, ...]
                   Higher weights for minority classes (e.g., [1.0, 8.0, 1.3])
            gamma: Focusing parameter. Higher gamma = more focus on hard examples.
                   gamma=0 is equivalent to cross-entropy loss.
                   Typical values: 0.5, 1.0, 2.0, 5.0
            reduction: 'none', 'mean', or 'sum'
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: [batch_size, num_classes] logits (before softmax)
            targets: [batch_size] class indices

        Returns:
            Focal loss value
        """
        num_classes = inputs.size(1)

        # Compute softmax probabilities
        p = F.softmax(inputs, dim=1)

        # Get probability of target class
        ce_loss = F.cross_entropy(
            inputs, targets,
            reduction='none',
            label_smoothing=self.label_smoothing
        )

        # Get p_t (probability of correct class)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal weight to cross-entropy loss
        focal_loss = focal_weight * ce_loss

        # Apply class weights (alpha)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples.

    From: "Class-Balanced Loss Based on Effective Number of Samples" (Cui et al., 2019)

    Effective number: E_n = (1 - beta^n) / (1 - beta)
    Weight: w_i = (1 - beta) / (1 - beta^n_i)
    """

    def __init__(
        self,
        samples_per_class: List[int],
        beta: float = 0.9999,
        gamma: float = 2.0,
        loss_type: str = 'focal'
    ):
        """
        Initialize Class-Balanced Loss.

        Args:
            samples_per_class: Number of samples for each class [n_0, n_1, n_2, ...]
            beta: Hyperparameter for effective number calculation (0.9, 0.99, 0.999, 0.9999)
            gamma: Focal loss gamma (only used if loss_type='focal')
            loss_type: 'focal', 'softmax', or 'sigmoid'
        """
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type

        # Compute effective number and weights
        effective_num = 1.0 - torch.pow(torch.tensor(beta), torch.tensor(samples_per_class, dtype=torch.float32))
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(samples_per_class)  # Normalize

        self.register_buffer('weights', weights)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute class-balanced loss.

        Args:
            inputs: [batch_size, num_classes] logits
            targets: [batch_size] class indices

        Returns:
            Class-balanced loss value
        """
        if self.loss_type == 'focal':
            return self._focal_loss(inputs, targets)
        elif self.loss_type == 'softmax':
            return F.cross_entropy(inputs, targets, weight=self.weights)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def _focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss with class-balanced weights."""
        p = F.softmax(inputs, dim=1)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply class-balanced weights
        weights_t = self.weights.gather(0, targets)

        return (weights_t * focal_weight * ce_loss).mean()


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    From: "Supervised Contrastive Learning" (Khosla et al., 2020)

    Pushes samples of the same class closer together while
    pushing samples of different classes apart.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = 'all',
        base_temperature: float = 0.07
    ):
        """
        Initialize Supervised Contrastive Loss.

        Args:
            temperature: Temperature for scaling (lower = sharper distribution)
            contrast_mode: 'one' or 'all'
            base_temperature: Base temperature for scaling
        """
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            features: [batch_size, feature_dim] normalized features
            labels: [batch_size] class labels
            mask: [batch_size, batch_size] optional contrastive mask

        Returns:
            Supervised contrastive loss
        """
        device = features.device
        batch_size = features.shape[0]

        # Normalize features
        features = F.normalize(features, dim=1)

        # Create mask based on labels
        labels = labels.contiguous().view(-1, 1)
        if mask is None:
            mask = torch.eq(labels, labels.T).float().to(device)

        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Create mask to exclude self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Compute mean of log-likelihood over positive samples
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int,
    method: str = 'inverse'
) -> torch.Tensor:
    """
    Compute class weights based on label distribution.

    Args:
        labels: Tensor of class labels
        num_classes: Number of classes
        method: 'inverse' (1/freq), 'sqrt_inverse' (1/sqrt(freq)),
                'effective' (effective number based)

    Returns:
        Tensor of class weights
    """
    # Count samples per class
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)  # Avoid division by zero

    if method == 'inverse':
        weights = 1.0 / counts
    elif method == 'sqrt_inverse':
        weights = 1.0 / torch.sqrt(counts)
    elif method == 'effective':
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / effective_num
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize so that weights sum to num_classes
    weights = weights / weights.sum() * num_classes

    return weights
