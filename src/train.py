"""
Training Script for CDAN Model
Main entry point for training the multimodal sentiment analysis model.
Enhanced with: warmup scheduler, mixup augmentation, EMA, early stopping.
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR
from tqdm import tqdm
import copy
import math

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from systems.cdan_model import build_cdan_model
from data.datamodule import build_datamodule
from data.augmentation import MixupAugmentation, mixup_criterion
from utils.metrics import MetricsCalculator, AverageMeter, ProgressTracker
from utils.seed import set_seed
from utils.logging import ExperimentLogger, log_system_info


class EMAModel:
    """
    Exponential Moving Average (EMA) for model weights.
    Improves generalization by maintaining a smoothed version of model weights.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Initialize EMA model.

        Args:
            model: The model to track
            decay: EMA decay factor (higher = smoother)
        """
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        """Update EMA weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )

    def apply_shadow(self, model: nn.Module):
        """Apply EMA weights to model (for evaluation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        """Restore original weights after evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Stops training if validation metric doesn't improve.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'max'
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum improvement to count as progress
            mode: 'max' for metrics to maximize (F1), 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01
):
    """
    Create cosine schedule with linear warmup.

    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of initial LR

    Returns:
        LR scheduler
    """
    def lr_lambda(current_step: int):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


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

    # Training (defaults=None to use config file values)
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate for head')
    parser.add_argument('--clip-lr', type=float, default=None,
                        help='Learning rate for CLIP encoders')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Weight decay')
    parser.add_argument('--freeze-epochs', type=int, default=None,
                        help='Number of epochs to freeze CLIP')
    parser.add_argument('--warmup-epochs', type=int, default=None,
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
    parser.add_argument('--gradient-clip', type=float, default=None,
                        help='Gradient clipping norm')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--save-freq', type=int, default=None,
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
    gradient_clip=1.0,
    scheduler=None,
    use_mixup=False,
    mixup_augmentation=None,
    ema_model=None,
    gradient_accumulation_steps=1
):
    """
    Train for one epoch with advanced techniques.

    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
        logger: Experiment logger
        gradient_clip: Gradient clipping threshold
        scheduler: Learning rate scheduler (for step-level scheduling)
        use_mixup: Whether to use mixup augmentation
        mixup_augmentation: Mixup augmentation instance
        ema_model: EMA model for weight averaging
        gradient_accumulation_steps: Number of steps to accumulate gradients
    """
    model.train()

    loss_meter = AverageMeter('Loss', ':.4f')
    acc_meter = AverageMeter('Acc', ':.2f')

    metrics_calc = MetricsCalculator(num_classes=model.num_classes)

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    optimizer.zero_grad()
    accumulation_counter = 0

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)

        # BERT inputs for dual encoders (CDAN 2025)
        bert_input_ids = batch.get('bert_input_ids')
        bert_attention_mask = batch.get('bert_attention_mask')
        if bert_input_ids is not None:
            bert_input_ids = bert_input_ids.to(device)
            bert_attention_mask = bert_attention_mask.to(device)

        # Apply mixup if enabled
        if use_mixup and mixup_augmentation is not None:
            batch_dict = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'pixel_values': pixel_values,
                'labels': labels
            }
            mixed_batch, labels_a, labels_b, lam = mixup_augmentation(batch_dict)

            # Forward with mixed inputs
            outputs = model(
                input_ids=mixed_batch['input_ids'],
                attention_mask=mixed_batch['attention_mask'],
                pixel_values=mixed_batch['pixel_values'],
                labels=None  # Compute loss manually for mixup
            )

            logits = outputs['logits']

            # Mixup loss
            loss = mixup_criterion(
                F.cross_entropy,
                logits,
                labels_a,
                labels_b,
                lam
            )

            # Add auxiliary loss if present
            if 'aux_loss' in outputs:
                aux_weight = model.aux_loss_weight if hasattr(model, 'aux_loss_weight') else 0.1
                loss = loss + aux_weight * outputs['aux_loss']
        else:
            # Standard forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                bert_input_ids=bert_input_ids,
                bert_attention_mask=bert_attention_mask
            )

            loss = outputs['loss']
            logits = outputs['logits']

        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps

        # Backward
        loss.backward()

        accumulation_counter += 1

        # Update weights after accumulation steps
        if accumulation_counter >= gradient_accumulation_steps:
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()
            optimizer.zero_grad()
            accumulation_counter = 0

            # Update EMA model
            if ema_model is not None:
                ema_model.update(model)

            # Step-level scheduler update
            if scheduler is not None and hasattr(scheduler, 'step_batch'):
                scheduler.step()

        # Metrics (use original labels for accuracy calculation)
        preds = torch.argmax(logits, dim=-1)
        if use_mixup and mixup_augmentation is not None:
            # For mixup, use weighted accuracy
            acc = lam * (preds == labels_a).float().mean() + \
                  (1 - lam) * (preds == labels_b).float().mean()
            metrics_calc.update(preds, labels_a, torch.softmax(logits, dim=-1))
        else:
            acc = (preds == labels).float().mean()
            metrics_calc.update(preds, labels, torch.softmax(logits, dim=-1))

        # Rescale loss for logging
        loss_meter.update(loss.item() * gradient_accumulation_steps, input_ids.size(0))
        acc_meter.update(acc.item() * 100, input_ids.size(0))

        # Update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.2f}%',
            'lr': f'{current_lr:.2e}'
        })

    # Handle any remaining accumulated gradients
    if accumulation_counter > 0:
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        optimizer.zero_grad()
        if ema_model is not None:
            ema_model.update(model)

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

            # BERT inputs for dual encoders (CDAN 2025)
            bert_input_ids = batch.get('bert_input_ids')
            bert_attention_mask = batch.get('bert_attention_mask')
            if bert_input_ids is not None:
                bert_input_ids = bert_input_ids.to(device)
                bert_attention_mask = bert_attention_mask.to(device)

            # Forward
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                bert_input_ids=bert_input_ids,
                bert_attention_mask=bert_attention_mask
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
    """Main training function with advanced techniques."""
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

    # Calculate total training steps for warmup scheduler
    num_training_steps = len(train_loader) * config['epochs']
    warmup_epochs = config.get('warmup_epochs', 3)
    num_warmup_steps = len(train_loader) * warmup_epochs

    # Scheduler
    scheduler_type = config.get('scheduler', 'cosine')
    if scheduler_type == 'cosine_warmup':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=0.01
        )
        scheduler.step_batch = True  # Mark for step-level updates
        exp_logger.log(f"Using cosine scheduler with {warmup_epochs} epochs warmup")
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
    elif scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    else:
        scheduler = None

    # Initialize EMA model
    ema_model = None
    if config.get('use_ema', False):
        ema_decay = config.get('ema_decay', 0.999)
        ema_model = EMAModel(model, decay=ema_decay)
        exp_logger.log(f"Using EMA with decay={ema_decay}")

    # Initialize mixup augmentation
    mixup_augmentation = None
    use_mixup = config.get('use_mixup', False)
    if use_mixup:
        mixup_alpha = config.get('mixup_alpha', 0.2)
        mixup_augmentation = MixupAugmentation(alpha=mixup_alpha)
        exp_logger.log(f"Using mixup augmentation with alpha={mixup_alpha}")

    # Initialize early stopping
    early_stopping = None
    if config.get('early_stopping', False):
        patience = config.get('patience', 10)
        min_delta = config.get('min_delta', 0.001)
        early_stopping = EarlyStopping(patience=patience, min_delta=min_delta, mode='max')
        exp_logger.log(f"Using early stopping with patience={patience}")

    # Gradient accumulation
    gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
    if config.get('use_gradient_accumulation', False) and gradient_accumulation_steps > 1:
        exp_logger.log(f"Using gradient accumulation with {gradient_accumulation_steps} steps")
        exp_logger.log(f"Effective batch size: {config['batch_size'] * gradient_accumulation_steps}")

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
    exp_logger.log(f"Configuration highlights:")
    exp_logger.log(f"  - Cross-attention layers: {config.get('cross_attn_layers', 2)}")
    exp_logger.log(f"  - Gate type: {config.get('gate_type', 'simple')}")
    exp_logger.log(f"  - Label smoothing: {config.get('label_smoothing', 0.0)}")
    exp_logger.log(f"  - Data augmentation: {config.get('augment_train', False)}")
    exp_logger.log(f"  - Mixup: {use_mixup}")
    exp_logger.log(f"  - EMA: {config.get('use_ema', False)}")
    exp_logger.log(f"  - Dual Encoders (CDAN 2025): {config.get('use_dual_encoders', False)}")
    if config.get('use_dual_encoders', False):
        exp_logger.log(f"    - BERT model: {config.get('bert_model_name', 'bert-base-uncased')}")
        exp_logger.log(f"    - ResNet model: {config.get('resnet_model', 'resnet50')}")

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
            gradient_clip=config['gradient_clip'],
            scheduler=scheduler if scheduler_type == 'cosine_warmup' else None,
            use_mixup=use_mixup,
            mixup_augmentation=mixup_augmentation,
            ema_model=ema_model,
            gradient_accumulation_steps=gradient_accumulation_steps if config.get('use_gradient_accumulation', False) else 1
        )

        # Validate (with EMA weights if available)
        if ema_model is not None:
            ema_model.apply_shadow(model)

        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            device=device,
            epoch=epoch,
            logger=exp_logger
        )

        if ema_model is not None:
            ema_model.restore(model)

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

        # Scheduler step (for non-step-level schedulers)
        if scheduler is not None and scheduler_type != 'cosine_warmup':
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['f1'])
            else:
                scheduler.step()

        # Save checkpoint
        is_best = val_metrics['f1'] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics['f1']
            exp_logger.log(f"New best model! Val F1: {best_val_f1:.4f}")

        save_best_only = config.get('save_best_only', False)
        if is_best or (not save_best_only and epoch % config['save_freq'] == 0):
            # Save with EMA weights if available
            if ema_model is not None:
                ema_model.apply_shadow(model)

            exp_logger.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={'val_f1': val_metrics['f1']},
                is_best=is_best
            )

            if ema_model is not None:
                ema_model.restore(model)

        # Check early stopping
        if early_stopping is not None:
            if early_stopping(val_metrics['f1']):
                exp_logger.log(f"Early stopping triggered at epoch {epoch}")
                break

    # Finalize
    exp_logger.log(f"\nTraining completed!")
    exp_logger.log(f"Best validation F1: {best_val_f1:.4f}")

    # Save training history
    tracker.save(os.path.join(exp_logger.log_dir, 'training_history.json'))
    exp_logger.finalize()


if __name__ == '__main__':
    main()
