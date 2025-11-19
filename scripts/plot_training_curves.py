#!/usr/bin/env python3
"""
Plot training curves from experiment logs.

Usage:
    python scripts/plot_training_curves.py --log-dir logs/mvsa_multiple_real_20251118_172701
"""

import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import numpy as np


def load_metrics(log_dir):
    """
    Load metrics from log directory.

    Looks for metrics.json or parses log file.
    """
    log_dir = Path(log_dir)

    # Try to load metrics.json
    metrics_file = log_dir / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            return json.load(f)

    # Otherwise parse the log file
    log_file = None
    for f in log_dir.glob('*.log'):
        log_file = f
        break

    if log_file is None:
        raise FileNotFoundError(f"No metrics.json or .log file found in {log_dir}")

    return parse_log_file(log_file)


def parse_log_file(log_file):
    """Parse metrics from log file."""
    metrics = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }

    with open(log_file, 'r') as f:
        for line in f:
            # Parse training metrics
            if '| train_loss:' in line and '| train_acc:' in line:
                parts = line.split('|')
                epoch = None
                train_loss = None
                train_acc = None

                for part in parts:
                    part = part.strip()
                    if part.startswith('Step '):
                        epoch = int(part.split()[1])
                    elif part.startswith('train_loss:'):
                        train_loss = float(part.split(':')[1].strip())
                    elif part.startswith('train_acc:'):
                        train_acc = float(part.split(':')[1].strip())

                if epoch is not None and train_loss is not None:
                    metrics['epochs'].append(epoch)
                    metrics['train_loss'].append(train_loss)
                    metrics['train_acc'].append(train_acc)

            # Parse validation metrics
            if '| val_loss:' in line:
                parts = line.split('|')
                val_loss = None
                val_acc = None
                val_f1 = None

                for part in parts:
                    part = part.strip()
                    if part.startswith('val_loss:'):
                        val_loss = float(part.split(':')[1].strip())
                    elif part.startswith('val_accuracy:'):
                        val_acc = float(part.split(':')[1].strip())
                    elif part.startswith('val_f1:'):
                        val_f1 = float(part.split(':')[1].strip())

                if val_loss is not None:
                    metrics['val_loss'].append(val_loss)
                    metrics['val_acc'].append(val_acc if val_acc else 0.0)
                    metrics['val_f1'].append(val_f1 if val_f1 else 0.0)

    return metrics


def plot_training_curves(metrics, output_dir):
    """
    Create comprehensive training plots.

    Args:
        metrics: Dictionary with training metrics
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = metrics['epochs']

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {
        'train': '#1f77b4',  # Blue
        'val': '#ff7f0e',    # Orange
        'accent': '#2ca02c'  # Green
    }

    # Figure 1: Loss curves (Train + Val)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(epochs, metrics['train_loss'],
            label='Training Loss',
            color=colors['train'],
            linewidth=2,
            marker='o',
            markersize=4,
            markevery=5)

    ax.plot(epochs, metrics['val_loss'],
            label='Validation Loss',
            color=colors['val'],
            linewidth=2,
            marker='s',
            markersize=4,
            markevery=5)

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    # Add annotations for best validation loss
    best_val_epoch = epochs[np.argmin(metrics['val_loss'])]
    best_val_loss = min(metrics['val_loss'])
    ax.annotate(f'Best: {best_val_loss:.4f}',
                xy=(best_val_epoch, best_val_loss),
                xytext=(10, 20),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'loss_curves.png'}")
    plt.close()

    # Figure 2: Accuracy curves (Train + Val)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Convert to percentage
    train_acc_pct = [acc * 100 for acc in metrics['train_acc']]
    val_acc_pct = [acc * 100 for acc in metrics['val_acc']]

    ax.plot(epochs, train_acc_pct,
            label='Training Accuracy',
            color=colors['train'],
            linewidth=2,
            marker='o',
            markersize=4,
            markevery=5)

    ax.plot(epochs, val_acc_pct,
            label='Validation Accuracy',
            color=colors['val'],
            linewidth=2,
            marker='s',
            markersize=4,
            markevery=5)

    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    # Add annotations for best validation accuracy
    best_val_acc_epoch = epochs[np.argmax(val_acc_pct)]
    best_val_acc = max(val_acc_pct)
    ax.annotate(f'Best: {best_val_acc:.2f}%',
                xy=(best_val_acc_epoch, best_val_acc),
                xytext=(10, -30),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_curves.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'accuracy_curves.png'}")
    plt.close()

    # Figure 3: Combined plot (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Training Loss
    axes[0, 0].plot(epochs, metrics['train_loss'],
                    color=colors['train'],
                    linewidth=2,
                    marker='o',
                    markersize=3)
    axes[0, 0].set_title('Training Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: Validation Loss
    axes[0, 1].plot(epochs, metrics['val_loss'],
                    color=colors['val'],
                    linewidth=2,
                    marker='s',
                    markersize=3)
    axes[0, 1].set_title('Validation Loss', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Training Accuracy
    axes[1, 0].plot(epochs, train_acc_pct,
                    color=colors['train'],
                    linewidth=2,
                    marker='o',
                    markersize=3)
    axes[1, 0].set_title('Training Accuracy', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_ylim([0, 100])
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: Validation Accuracy
    axes[1, 1].plot(epochs, val_acc_pct,
                    color=colors['val'],
                    linewidth=2,
                    marker='s',
                    markersize=3)
    axes[1, 1].set_title('Validation Accuracy', fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy (%)')
    axes[1, 1].set_ylim([0, 100])
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Training Progress Overview',
                 fontsize=16,
                 fontweight='bold',
                 y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_overview.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'training_overview.png'}")
    plt.close()

    # Figure 4: Validation F1 Score
    if metrics['val_f1'] and any(f > 0 for f in metrics['val_f1']):
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(epochs, metrics['val_f1'],
                label='Validation F1',
                color=colors['accent'],
                linewidth=2,
                marker='D',
                markersize=4,
                markevery=5)

        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('Validation F1 Score', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # Add annotations for best F1
        best_f1_epoch = epochs[np.argmax(metrics['val_f1'])]
        best_f1 = max(metrics['val_f1'])
        ax.annotate(f'Best: {best_f1:.4f}',
                    xy=(best_f1_epoch, best_f1),
                    xytext=(10, -30),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightblue', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        plt.tight_layout()
        plt.savefig(output_dir / 'f1_curve.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir / 'f1_curve.png'}")
        plt.close()

    # Print summary statistics
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Total Epochs: {len(epochs)}")
    print(f"\nFinal Metrics:")
    print(f"  Train Loss: {metrics['train_loss'][-1]:.4f}")
    print(f"  Train Acc:  {metrics['train_acc'][-1]*100:.2f}%")
    print(f"  Val Loss:   {metrics['val_loss'][-1]:.4f}")
    print(f"  Val Acc:    {metrics['val_acc'][-1]*100:.2f}%")
    if metrics['val_f1'] and metrics['val_f1'][-1] > 0:
        print(f"  Val F1:     {metrics['val_f1'][-1]:.4f}")
    print(f"\nBest Metrics:")
    print(f"  Best Val Loss: {min(metrics['val_loss']):.4f} (Epoch {epochs[np.argmin(metrics['val_loss'])]})")
    print(f"  Best Val Acc:  {max(metrics['val_acc'])*100:.2f}% (Epoch {epochs[np.argmax(metrics['val_acc'])]})")
    if metrics['val_f1'] and any(f > 0 for f in metrics['val_f1']):
        print(f"  Best Val F1:   {max(metrics['val_f1']):.4f} (Epoch {epochs[np.argmax(metrics['val_f1'])]})")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plot training curves from experiment logs')
    parser.add_argument('--log-dir', type=str, required=True,
                        help='Path to log directory (e.g., logs/mvsa_multiple_real_20251118_172701)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: same as log-dir)')

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.log_dir

    print(f"Loading metrics from: {args.log_dir}")
    metrics = load_metrics(args.log_dir)

    print(f"Generating plots...")
    plot_training_curves(metrics, args.output_dir)

    print(f"\n✓ All plots saved to: {args.output_dir}")
    print("\nGenerated files:")
    print(f"  - loss_curves.png         (Train + Val Loss)")
    print(f"  - accuracy_curves.png     (Train + Val Accuracy)")
    print(f"  - training_overview.png   (2x2 Grid Overview)")
    print(f"  - f1_curve.png           (Validation F1)")


if __name__ == '__main__':
    main()
