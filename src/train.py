"""
Training Script for CDAN Model
Main entry point for training the multimodal sentiment analysis model.
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from systems.cdan_model import build_cdan_model
from data.datamodule import build_datamodule
from utils.metrics import MetricsCalculator, AverageMeter, ProgressTracker
from utils.seed import set_seed
from utils.logging import ExperimentLogger, log_system_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train CDAN Model for Multimodal Sentiment Analysis')

    # Configuration
    parser.add_argument('--config', type=str, default='configs/cdan.yaml',
                        help='Path to configuration file')
    parser.add_argument('--experiment-name', type=str, default='cdan_mvsa',
                        help='Experiment name')

    # Data
    parser.add_argument('--data-dir', type=str, default='data/processed',
                        help='Data directory')
    parser.add_argument('--image-dir', type=str, default='data/raw/images',
                        help='Image directory')
    parser.add_argument('--dataset-type', type=str, default='single',
                        choices=['single', 'multiple'], help='MVSA dataset type')

    # Training
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for head')
    parser.add_argument('--clip-lr', type=float, default=1e-5,
                        help='Learning rate for CLIP encoders')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--freeze-epochs', type=int, default=3,
                        help='Number of epochs to freeze CLIP')
    parser.add_argument('--warmup-epochs', type=int, default=2,
                        help='Number of warmup epochs')

    # Model
    parser.add_argument('--clip-model', type=str, default='openai/clip-vit-base-patch32',
                        help='CLIP model name')
    parser.add_argument('--num-classes', type=int, default=3,
                        help='Number of sentiment classes')

    # Optimization
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'step'], help='LR scheduler')
    parser.add_argument('--gradient-clip', type=float, default=1.0,
                        help='Gradient clipping norm')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--save-freq', type=int, default=5,
                        help='Checkpoint save frequency (epochs)')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Log directory')

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    return {}


def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    epoch,
    logger,
    gradient_clip=1.0
):
    """Train for one epoch."""
    model.train()

    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.2f')

    metrics_calc = MetricsCalculator(num_classes=model.num_classes)

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    for batch in pbar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        # Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels
        )

        loss = outputs['loss']
        logits = outputs['logits']

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        # Metrics
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean()

        loss_meter.update(loss.item(), input_ids.size(0))
        acc_meter.update(acc.item() * 100, input_ids.size(0))

        metrics_calc.update(preds, labels, torch.softmax(logits, dim=-1))

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.2f}%'
        })

    # Compute metrics
    metrics = metrics_calc.compute()
    metrics['loss'] = loss_meter.avg
    metrics['acc'] = acc_meter.avg

    return metrics


def validate(model, dataloader, device, epoch, logger):
    """Validate model."""
    model.eval()

    loss_meter = AverageMeter('Loss', ':.4f')
    metrics_calc = MetricsCalculator(num_classes=model.num_classes)

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')

    with torch.no_grad():
        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            # Forward
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels
            )

            loss = outputs['loss']
            logits = outputs['logits']

            # Metrics
            preds = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)

            loss_meter.update(loss.item(), input_ids.size(0))
            metrics_calc.update(preds, labels, probs)

            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})

    # Compute metrics
    metrics = metrics_calc.compute()
    metrics['loss'] = loss_meter.avg

    return metrics


def main():
    """Main training function."""
    args = parse_args()

    # Load config file if exists
    config = load_config(args.config)

    # Override config with command line args
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    # Set seed
    set_seed(config['seed'])

    # Log system info
    log_system_info()

    # Setup logger
    exp_logger = ExperimentLogger(
        experiment_name=config['experiment_name'],
        log_dir=config['log_dir'],
        checkpoint_dir=config['checkpoint_dir']
    )
    exp_logger.log_config(config)

    # Device
    device = torch.device(config['device'])
    exp_logger.log(f"Using device: {device}")

    # Build data module
    exp_logger.log("Building data module...")
    datamodule = build_datamodule(config)
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    exp_logger.log(f"Train batches: {len(train_loader)}")
    exp_logger.log(f"Val batches: {len(val_loader)}")

    # Build model
    exp_logger.log("Building model...")
    model = build_cdan_model(config)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    exp_logger.log(f"Total parameters: {total_params:,}")
    exp_logger.log(f"Trainable parameters: {trainable_params:,}")

    # Optimizer
    # Separate parameter groups for CLIP and head
    clip_params = []
    head_params = []

    for name, param in model.named_parameters():
        if 'text_encoder' in name or 'image_encoder' in name:
            clip_params.append(param)
        else:
            head_params.append(param)

    if config['optimizer'] == 'adamw':
        optimizer = AdamW([
            {'params': head_params, 'lr': config['lr']},
            {'params': clip_params, 'lr': config['clip_lr']}
        ], weight_decay=config['weight_decay'])
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

    # Scheduler
    if config['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    elif config['scheduler'] == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    else:
        scheduler = None

    # Resume from checkpoint
    start_epoch = 1
    if config['resume']:
        exp_logger.log(f"Resuming from checkpoint: {config['resume']}")
        checkpoint = exp_logger.load_checkpoint(config['resume'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # Training loop
    exp_logger.log("Starting training...")
    tracker = ProgressTracker()
    best_val_f1 = 0.0

    for epoch in range(start_epoch, config['epochs'] + 1):
        exp_logger.log(f"\n{'='*50}")
        exp_logger.log(f"Epoch {epoch}/{config['epochs']}")
        exp_logger.log(f"{'='*50}")

        # Freeze/unfreeze CLIP
        if epoch == 1:
            model.freeze_clip()
            exp_logger.log("CLIP encoders frozen")
        elif epoch == config['freeze_epochs'] + 1:
            model.unfreeze_clip_last_block()
            exp_logger.log("CLIP last blocks unfrozen")

        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            logger=exp_logger,
            gradient_clip=config['gradient_clip']
        )

        # Validate
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            device=device,
            epoch=epoch,
            logger=exp_logger
        )

        # Log metrics
        exp_logger.log_metrics(train_metrics, epoch, prefix='train_')
        exp_logger.log_metrics(val_metrics, epoch, prefix='val_')

        # Update tracker
        tracker.update(epoch, {
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_f1': val_metrics['f1'],
            'learning_rates': optimizer.param_groups[0]['lr']
        })

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['f1'])
            else:
                scheduler.step()

        # Save checkpoint
        is_best = val_metrics['f1'] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics['f1']
            exp_logger.log(f"New best model! Val F1: {best_val_f1:.4f}")

        if epoch % config['save_freq'] == 0 or is_best:
            exp_logger.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={'val_f1': val_metrics['f1']},
                is_best=is_best
            )

    # Finalize
    exp_logger.log(f"\nTraining completed!")
    exp_logger.log(f"Best validation F1: {best_val_f1:.4f}")

    # Save training history
    tracker.save(os.path.join(exp_logger.log_dir, 'training_history.json'))
    exp_logger.finalize()


if __name__ == '__main__':
    main()
