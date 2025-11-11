"""
Evaluation Script for CDAN Model
Evaluate trained model and generate detailed metrics, visualizations, and qualitative analysis.
"""

import argparse
import os
import sys
import json
from pathlib import Path
import yaml
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from systems.cdan_model import build_cdan_model
from data.datamodule import build_datamodule
from utils.metrics import MetricsCalculator
from utils.seed import set_seed
from utils.logging import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate CDAN Model')

    # Model and data
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/cdan.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Data directory')
    parser.add_argument('--image-dir', type=str, default='data/raw/images',
                        help='Image directory')
    parser.add_argument('--test-file', type=str, default=None,
                        help='Test data file (optional)')

    # Output
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save predictions to file')
    parser.add_argument('--visualize-attention', action='store_true',
                        help='Visualize attention weights')
    parser.add_argument('--num-viz-samples', type=int, default=10,
                        help='Number of samples to visualize')

    # Misc
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    return {}


def evaluate_model(model, dataloader, device, return_attention=False):
    """
    Evaluate model on dataloader.

    Args:
        model: Model to evaluate
        dataloader: DataLoader
        device: Device
        return_attention: Whether to return attention weights

    Returns:
        dict with metrics and predictions
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_sample_ids = []
    all_attentions = [] if return_attention else None

    pbar = tqdm(dataloader, desc='Evaluating')

    with torch.no_grad():
        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            sample_ids = batch['sample_ids']

            # Forward
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=None,
                return_attention=return_attention
            )

            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            # Store results
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_sample_ids.extend(sample_ids)

            if return_attention and 'attention_weights' in outputs:
                all_attentions.append(outputs['attention_weights'])

    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    results = {
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'sample_ids': all_sample_ids
    }

    if return_attention:
        results['attentions'] = all_attentions

    return results


def save_predictions(results, output_dir, class_names):
    """Save predictions to CSV file."""
    import pandas as pd

    df = pd.DataFrame({
        'sample_id': results['sample_ids'],
        'true_label': [class_names[i] for i in results['labels']],
        'pred_label': [class_names[i] for i in results['predictions']],
        'confidence': [results['probabilities'][i][pred] for i, pred in enumerate(results['predictions'])]
    })

    # Add probability columns
    for i, class_name in enumerate(class_names):
        df[f'prob_{class_name}'] = results['probabilities'][:, i]

    # Add correctness
    df['correct'] = df['true_label'] == df['pred_label']

    # Save
    output_file = os.path.join(output_dir, 'predictions.csv')
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    return df


def plot_confusion_matrix(cm, class_names, output_dir):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    output_file = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_file}")


def plot_per_class_metrics(metrics, output_dir):
    """Plot per-class metrics."""
    per_class = metrics['per_class']
    class_names = list(per_class.keys())

    precision = [per_class[c]['precision'] for c in class_names]
    recall = [per_class[c]['recall'] for c in class_names]
    f1 = [per_class[c]['f1'] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'per_class_metrics.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class metrics plot saved to {output_file}")


def analyze_errors(predictions_df, output_dir):
    """Analyze prediction errors."""
    errors = predictions_df[predictions_df['correct'] == False]

    print(f"\nError Analysis:")
    print(f"Total errors: {len(errors)} / {len(predictions_df)} "
          f"({len(errors) / len(predictions_df) * 100:.2f}%)")

    # Error breakdown by true label
    print("\nErrors by true label:")
    error_by_true = errors.groupby('true_label').size()
    for label, count in error_by_true.items():
        total = len(predictions_df[predictions_df['true_label'] == label])
        print(f"  {label}: {count}/{total} ({count/total*100:.2f}%)")

    # Most common misclassifications
    print("\nMost common misclassifications:")
    misclass = errors.groupby(['true_label', 'pred_label']).size().sort_values(ascending=False)
    for (true_label, pred_label), count in misclass.head(5).items():
        print(f"  {true_label} -> {pred_label}: {count}")

    # Low confidence predictions
    low_conf = predictions_df[predictions_df['confidence'] < 0.5].sort_values('confidence')
    if len(low_conf) > 0:
        print(f"\nLow confidence predictions: {len(low_conf)}")
        print(low_conf[['sample_id', 'true_label', 'pred_label', 'confidence']].head(10))

    # Save error analysis
    error_analysis = {
        'total_errors': int(len(errors)),
        'error_rate': float(len(errors) / len(predictions_df)),
        'errors_by_true_label': error_by_true.to_dict(),
        'most_common_misclassifications': [
            {'true_label': true_label, 'pred_label': pred_label, 'count': int(count)}
            for (true_label, pred_label), count in misclass.head(10).items()
        ]
    }

    with open(os.path.join(output_dir, 'error_analysis.json'), 'w') as f:
        json.dump(error_analysis, f, indent=2)


def main():
    """Main evaluation function."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup logging
    logger = setup_logging(log_dir=args.output_dir, experiment_name='evaluation')
    logger.info("Starting evaluation...")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = load_config(args.config)
    config.update(vars(args))

    # Device
    device = torch.device(config['device'])
    logger.info(f"Using device: {device}")

    # Build data module
    logger.info("Loading data...")
    if args.test_file:
        config['test_file'] = args.test_file

    datamodule = build_datamodule(config)
    datamodule.setup('test')
    test_loader = datamodule.test_dataloader()

    logger.info(f"Test samples: {len(datamodule.test_dataset)}")
    logger.info(f"Test batches: {len(test_loader)}")

    # Build model
    logger.info("Building model...")
    model = build_cdan_model(config)

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    if 'epoch' in checkpoint:
        logger.info(f"Checkpoint from epoch {checkpoint['epoch']}")

    # Evaluate
    logger.info("Evaluating model...")
    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        return_attention=args.visualize_attention
    )

    # Compute metrics
    logger.info("Computing metrics...")
    class_names = ['positive', 'negative', 'neutral']
    metrics_calc = MetricsCalculator(
        num_classes=len(class_names),
        class_names=class_names
    )

    metrics_calc.update(
        predictions=results['predictions'],
        labels=results['labels'],
        probabilities=results['probabilities']
    )

    metrics = metrics_calc.compute()

    # Print metrics
    metrics_calc.print_metrics(metrics)

    # Save metrics
    metrics_file = os.path.join(output_dir, 'metrics.json')
    metrics_calc.save_metrics(metrics_file, metrics)

    # Plot confusion matrix
    logger.info("Generating visualizations...")
    cm = metrics_calc.get_confusion_matrix()
    plot_confusion_matrix(cm, class_names, output_dir)

    # Plot per-class metrics
    plot_per_class_metrics(metrics, output_dir)

    # Save predictions
    if args.save_predictions:
        logger.info("Saving predictions...")
        predictions_df = save_predictions(results, output_dir, class_names)

        # Error analysis
        analyze_errors(predictions_df, output_dir)

    # Visualize attention (if requested)
    if args.visualize_attention and results.get('attentions'):
        logger.info("Visualizing attention maps...")
        # Note: Attention visualization requires additional implementation
        # This is a placeholder for the feature
        logger.info("Attention visualization not fully implemented yet")

    logger.info(f"\nEvaluation complete! Results saved to {output_dir}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")


if __name__ == '__main__':
    main()
