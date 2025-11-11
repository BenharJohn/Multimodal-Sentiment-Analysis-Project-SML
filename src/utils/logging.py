"""
Logging Utilities
Configure logging and experiment tracking.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    log_dir: str = 'logs',
    experiment_name: str = 'cdan',
    log_level: int = logging.INFO,
    console: bool = True,
    file: bool = True
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        log_dir: Directory for log files
        experiment_name: Name of the experiment
        log_level: Logging level
        console: Whether to log to console
        file: Whether to log to file

    Returns:
        Configured logger
    """
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(log_level)
    logger.handlers = []  # Clear existing handlers

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{experiment_name}_{timestamp}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    return logger


class ExperimentLogger:
    """Experiment logging and tracking."""

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = 'logs',
        checkpoint_dir: str = 'checkpoints',
        save_config: bool = True
    ):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for logs
            checkpoint_dir: Directory for checkpoints
            save_config: Whether to save configuration
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f'{experiment_name}_{self.timestamp}'

        # Create directories
        self.log_dir = Path(log_dir) / self.run_name
        self.checkpoint_dir = Path(checkpoint_dir) / self.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = setup_logging(
            log_dir=str(self.log_dir),
            experiment_name=experiment_name,
            console=True,
            file=True
        )

        self.logger.info(f"Experiment: {self.run_name}")
        self.logger.info(f"Log directory: {self.log_dir}")
        self.logger.info(f"Checkpoint directory: {self.checkpoint_dir}")

        # Metrics storage
        self.metrics = {}

    def log(self, message: str, level: int = logging.INFO):
        """Log a message."""
        self.logger.log(level, message)

    def log_config(self, config: dict):
        """Log configuration."""
        import json
        config_file = self.log_dir / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        self.logger.info(f"Configuration saved to {config_file}")

        # Also log to console
        self.logger.info("Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")

    def log_metrics(self, metrics: dict, step: Optional[int] = None, prefix: str = ''):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics
            step: Current step/epoch
            prefix: Prefix for metric names
        """
        step_str = f"Step {step} | " if step is not None else ""
        metrics_str = " | ".join([f"{prefix}{k}: {v:.4f}" if isinstance(v, float)
                                   else f"{prefix}{k}: {v}"
                                   for k, v in metrics.items()])
        self.logger.info(f"{step_str}{metrics_str}")

        # Store metrics
        if step is not None:
            if step not in self.metrics:
                self.metrics[step] = {}
            self.metrics[step].update({f"{prefix}{k}": v for k, v in metrics.items()})

    def save_checkpoint(
        self,
        model,
        optimizer,
        epoch: int,
        metrics: dict,
        is_best: bool = False,
        filename: Optional[str] = None
    ):
        """
        Save model checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Metrics dictionary
            is_best: Whether this is the best model
            filename: Custom filename (optional)
        """
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pth'

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': self.timestamp
        }

        import torch
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint dictionary
        """
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint

    def finalize(self):
        """Finalize experiment (save metrics, etc.)."""
        # Save all metrics
        import json
        metrics_file = self.log_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.logger.info(f"Metrics saved to {metrics_file}")
        self.logger.info("Experiment completed")


def log_system_info():
    """Log system and environment information."""
    import torch
    import platform

    info = {
        'Python version': sys.version,
        'PyTorch version': torch.__version__,
        'CUDA available': torch.cuda.is_available(),
        'CUDA version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
        'GPU count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'Platform': platform.platform(),
        'Processor': platform.processor()
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            info[f'GPU {i}'] = torch.cuda.get_device_name(i)

    print("\n" + "="*50)
    print("System Information")
    print("="*50)
    for key, value in info.items():
        print(f"{key}: {value}")
    print("="*50 + "\n")

    return info
