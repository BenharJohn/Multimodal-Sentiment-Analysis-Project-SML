"""
Evaluation Metrics for Sentiment Analysis
Implements accuracy, precision, recall, F1, confusion matrix, etc.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from typing import Dict, List, Tuple, Optional
import json


class MetricsCalculator:
    """Calculate and track evaluation metrics."""

    def __init__(
        self,
        num_classes: int = 3,
        class_names: List[str] = None,
        average: str = 'macro'
    ):
        """
        Initialize metrics calculator.

        Args:
            num_classes: Number of classes
            class_names: Names of classes (e.g., ['positive', 'negative', 'neutral'])
            average: Averaging method for metrics ('macro', 'micro', 'weighted')
        """
        self.num_classes = num_classes
        self.average = average

        if class_names is None:
            class_names = [f'class_{i}' for i in range(num_classes)]
        self.class_names = class_names

        # Storage for predictions and labels
        self.reset()

    def reset(self):
        """Reset stored predictions and labels."""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []

    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        probabilities: Optional[torch.Tensor] = None
    ):
        """
        Update with new predictions and labels.

        Args:
            predictions: [batch_size] predicted class indices
            labels: [batch_size] ground truth labels
            probabilities: [batch_size, num_classes] class probabilities (optional)
        """
        # Convert to numpy
        preds_np = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
        labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels

        self.all_preds.extend(preds_np.tolist())
        self.all_labels.extend(labels_np.tolist())

        if probabilities is not None:
            if torch.is_tensor(probabilities):
                probs_np = probabilities.detach().cpu().numpy()
            else:
                probs_np = probabilities
            self.all_probs.extend(probs_np.tolist())

    def compute(self) -> Dict:
        """
        Compute all metrics.

        Returns:
            dict with various metrics
        """
        if len(self.all_preds) == 0:
            return {}

        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)

        # Basic metrics
        acc = accuracy_score(labels, preds)

        # Precision, Recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            labels,
            preds,
            average=self.average,
            zero_division=0
        )

        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, per_class_support = \
            precision_recall_fscore_support(
                labels,
                preds,
                average=None,
                zero_division=0
            )

        # Confusion matrix
        cm = confusion_matrix(labels, preds)

        # Classification report
        report = classification_report(
            labels,
            preds,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )

        metrics = {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'macro_f1': float(f1) if self.average == 'macro' else float(
                precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)[2]
            ),
            'confusion_matrix': cm.tolist(),
            'per_class': {}
        }

        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            if i < len(per_class_precision):
                metrics['per_class'][class_name] = {
                    'precision': float(per_class_precision[i]),
                    'recall': float(per_class_recall[i]),
                    'f1': float(per_class_f1[i]),
                    'support': int(per_class_support[i])
                }

        # AUC if probabilities available
        if len(self.all_probs) > 0:
            probs = np.array(self.all_probs)
            try:
                if self.num_classes == 2:
                    auc = roc_auc_score(labels, probs[:, 1])
                    metrics['auc'] = float(auc)
                else:
                    auc = roc_auc_score(labels, probs, multi_class='ovr', average=self.average)
                    metrics['auc'] = float(auc)
            except Exception as e:
                print(f"Warning: Could not compute AUC: {e}")

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(self.all_labels, self.all_preds)

    def print_metrics(self, metrics: Optional[Dict] = None):
        """
        Print metrics in readable format.

        Args:
            metrics: Metrics dict (if None, will compute)
        """
        if metrics is None:
            metrics = self.compute()

        print("\n" + "="*50)
        print("Evaluation Metrics")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"Macro F1:  {metrics['macro_f1']:.4f}")

        if 'auc' in metrics:
            print(f"AUC:       {metrics['auc']:.4f}")

        print("\nPer-Class Metrics:")
        print("-"*50)
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"{class_name:12s} | "
                  f"P: {class_metrics['precision']:.4f} | "
                  f"R: {class_metrics['recall']:.4f} | "
                  f"F1: {class_metrics['f1']:.4f} | "
                  f"Support: {class_metrics['support']}")

        print("\nConfusion Matrix:")
        print("-"*50)
        cm = np.array(metrics['confusion_matrix'])
        # Print header
        print("           ", end="")
        for name in self.class_names:
            print(f"{name[:8]:>8s}", end=" ")
        print()
        # Print rows
        for i, name in enumerate(self.class_names):
            print(f"{name[:10]:10s}", end=" ")
            for j in range(len(self.class_names)):
                if i < cm.shape[0] and j < cm.shape[1]:
                    print(f"{cm[i, j]:8d}", end=" ")
                else:
                    print(f"{'0':8s}", end=" ")
            print()
        print("="*50 + "\n")

    def save_metrics(self, filepath: str, metrics: Optional[Dict] = None):
        """
        Save metrics to JSON file.

        Args:
            filepath: Output file path
            metrics: Metrics dict (if None, will compute)
        """
        if metrics is None:
            metrics = self.compute()

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to {filepath}")


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = '', fmt: str = ':f'):
        """
        Initialize meter.

        Args:
            name: Name of the metric
            fmt: Format string for printing
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update statistics.

        Args:
            val: Value to add
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """String representation."""
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressTracker:
    """Track training progress and metrics."""

    def __init__(self):
        """Initialize tracker."""
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'learning_rates': []
        }
        self.best_val_f1 = 0.0
        self.best_epoch = 0

    def update(self, epoch: int, metrics: Dict):
        """
        Update history with new metrics.

        Args:
            epoch: Current epoch
            metrics: Dictionary of metrics
        """
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

        # Check for best model
        if 'val_f1' in metrics:
            if metrics['val_f1'] > self.best_val_f1:
                self.best_val_f1 = metrics['val_f1']
                self.best_epoch = epoch

    def save(self, filepath: str):
        """Save history to JSON."""
        data = {
            'history': self.history,
            'best_val_f1': self.best_val_f1,
            'best_epoch': self.best_epoch
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load history from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.history = data['history']
        self.best_val_f1 = data['best_val_f1']
        self.best_epoch = data['best_epoch']
