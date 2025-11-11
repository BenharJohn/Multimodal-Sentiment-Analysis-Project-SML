"""
Visualization Utilities for CDAN Model
Generates plots for training curves, confusion matrices, and attention heatmaps.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import torch


def plot_training_curves(
    history: Dict[str, List],
    output_dir: str,
    metrics: List[str] = ['loss', 'acc', 'f1']
):
    """
    Plot training and validation curves.

    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss', etc.
        output_dir: Directory to save plots
        metrics: List of metrics to plot
    """
    os.makedirs(output_dir, exist_ok=True)

    for metric in metrics:
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'

        if train_key not in history or val_key not in history:
            continue

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(history[train_key]) + 1)

        plt.plot(epochs, history[train_key], 'b-', label=f'Train {metric}', linewidth=2)
        plt.plot(epochs, history[val_key], 'r-', label=f'Val {metric}', linewidth=2)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric.upper(), fontsize=12)
        plt.title(f'Training and Validation {metric.upper()}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_file = os.path.join(output_dir, f'{metric}_curve.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved {metric} curve to {output_file}")


def plot_confusion_matrix_detailed(
    cm: np.ndarray,
    class_names: List[str],
    output_dir: str,
    normalize: bool = False,
    title: str = 'Confusion Matrix'
):
    """
    Plot confusion matrix with detailed annotations.

    Args:
        cm: Confusion matrix array [num_classes, num_classes]
        class_names: List of class names
        output_dir: Directory to save plot
        normalize: Whether to normalize by row (true labels)
        title: Plot title
    """
    os.makedirs(output_dir, exist_ok=True)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count' if normalize else 'Count'},
        linewidths=0.5,
        linecolor='gray'
    )

    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    suffix = '_normalized' if normalize else ''
    output_file = os.path.join(output_dir, f'confusion_matrix{suffix}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved confusion matrix to {output_file}")


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    text_tokens: List[str],
    image_patch_indices: Optional[List[int]] = None,
    output_path: str = 'attention_heatmap.png',
    title: str = 'Cross-Attention Weights'
):
    """
    Visualize cross-attention weights between text and image.

    Args:
        attention_weights: [num_heads, text_len, image_patches] or [text_len, image_patches]
        text_tokens: List of text tokens
        image_patch_indices: List of patch indices (optional)
        output_path: Output file path
        title: Plot title
    """
    # Average over heads if needed
    if attention_weights.dim() == 3:
        attention_weights = attention_weights.mean(dim=0)

    # Convert to numpy
    attn_np = attention_weights.cpu().numpy()

    # Limit display size
    max_tokens = 20
    max_patches = 30
    if len(text_tokens) > max_tokens:
        text_tokens = text_tokens[:max_tokens]
        attn_np = attn_np[:max_tokens, :]
    if attn_np.shape[1] > max_patches:
        attn_np = attn_np[:, :max_patches]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot heatmap
    im = ax.imshow(attn_np, cmap='viridis', aspect='auto', interpolation='nearest')

    # Set ticks
    ax.set_xticks(np.arange(attn_np.shape[1]))
    ax.set_yticks(np.arange(len(text_tokens)))

    # Labels
    if image_patch_indices:
        ax.set_xticklabels([f'P{i}' for i in image_patch_indices[:attn_np.shape[1]]])
    else:
        ax.set_xticklabels([f'P{i}' for i in range(attn_np.shape[1])])
    ax.set_yticklabels(text_tokens)

    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20)

    # Title and labels
    ax.set_xlabel('Image Patches', fontsize=12, fontweight='bold')
    ax.set_ylabel('Text Tokens', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved attention heatmap to {output_path}")


def plot_per_class_metrics_comparison(
    metrics_dict: Dict[str, Dict],
    output_dir: str,
    filename: str = 'per_class_comparison.png'
):
    """
    Plot per-class precision, recall, F1 comparison.

    Args:
        metrics_dict: Dict with class names as keys and metric dicts as values
        output_dir: Directory to save plot
        filename: Output filename
    """
    os.makedirs(output_dir, exist_ok=True)

    class_names = list(metrics_dict.keys())
    precision = [metrics_dict[c]['precision'] for c in class_names]
    recall = [metrics_dict[c]['recall'] for c in class_names]
    f1 = [metrics_dict[c]['f1'] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.9, color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.9, color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.9, color='#e74c3c')

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    ax.set_xlabel('Class', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved per-class metrics to {output_path}")


def visualize_sample_predictions(
    images: List,
    texts: List[str],
    true_labels: List[str],
    pred_labels: List[str],
    probs: List[np.ndarray],
    output_dir: str,
    num_samples: int = 9
):
    """
    Visualize sample predictions with images, text, and predictions.

    Args:
        images: List of PIL Images or numpy arrays
        texts: List of text strings
        true_labels: List of true label strings
        pred_labels: List of predicted label strings
        probs: List of probability arrays [num_classes]
        output_dir: Directory to save plot
        num_samples: Number of samples to display
    """
    os.makedirs(output_dir, exist_ok=True)

    num_samples = min(num_samples, len(images))
    cols = 3
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for idx in range(num_samples):
        ax = axes[idx]

        # Display image
        if hasattr(images[idx], 'numpy'):
            img = images[idx].numpy()
        else:
            img = images[idx]

        ax.imshow(img)
        ax.axis('off')

        # Truncate text
        text_display = texts[idx][:50] + '...' if len(texts[idx]) > 50 else texts[idx]

        # Create title
        is_correct = true_labels[idx] == pred_labels[idx]
        color = 'green' if is_correct else 'red'
        confidence = np.max(probs[idx])

        title = f"True: {true_labels[idx]} | Pred: {pred_labels[idx]}\n"
        title += f"Conf: {confidence:.2f} | Text: {text_display}"

        ax.set_title(title, fontsize=9, color=color, fontweight='bold', wrap=True)

    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'sample_predictions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved sample predictions to {output_path}")


def plot_loss_components(
    history: Dict[str, List],
    output_dir: str
):
    """
    Plot individual loss components (CE loss, aux loss, total loss).

    Args:
        history: Dictionary with loss component histories
        output_dir: Directory to save plot
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Training losses
    if 'train_ce_loss' in history:
        epochs = range(1, len(history['train_ce_loss']) + 1)
        ax1.plot(epochs, history['train_ce_loss'], label='CE Loss', linewidth=2)
        if 'train_aux_loss' in history:
            ax1.plot(epochs, history['train_aux_loss'], label='Aux Loss', linewidth=2)
        if 'train_loss' in history:
            ax1.plot(epochs, history['train_loss'], label='Total Loss', linewidth=2, linestyle='--')

        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Loss Components', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

    # Validation losses
    if 'val_ce_loss' in history:
        epochs = range(1, len(history['val_ce_loss']) + 1)
        ax2.plot(epochs, history['val_ce_loss'], label='CE Loss', linewidth=2)
        if 'val_aux_loss' in history:
            ax2.plot(epochs, history['val_aux_loss'], label='Aux Loss', linewidth=2)
        if 'val_loss' in history:
            ax2.plot(epochs, history['val_loss'], label='Total Loss', linewidth=2, linestyle='--')

        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Validation Loss Components', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'loss_components.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved loss components plot to {output_path}")
