"""
Seed Setting Utilities
Ensure reproducibility across runs.
"""

import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic mode (slower but reproducible)
    """
    # Python random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

    if deterministic:
        # Enable deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # For PyTorch >= 1.8
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
    else:
        # For better performance
        torch.backends.cudnn.benchmark = True

    print(f"Random seed set to {seed}")
    if deterministic:
        print("Deterministic mode enabled (slower but reproducible)")
    else:
        print("Deterministic mode disabled (faster but may vary)")


def worker_init_fn(worker_id: int, seed: int = 42):
    """
    Initialize worker seed for DataLoader workers.

    Args:
        worker_id: Worker ID
        seed: Base random seed
    """
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
