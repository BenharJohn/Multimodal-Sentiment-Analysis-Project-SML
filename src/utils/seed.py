"""
Seed Setting Utilities
Ensure reproducibility across runs.
"""

import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42, deterministic: bool = False):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic mode (slower but reproducible)
                      Default is False to avoid CUBLAS errors on CUDA >= 10.2
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
        # Set CUBLAS workspace config for deterministic cuBLAS operations
        # Required for CUDA >= 10.2 when using deterministic algorithms
        if 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        # Enable deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # For PyTorch >= 1.8
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
    else:
        # For better performance (still reproducible for most operations)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")
    if deterministic:
        print("Deterministic mode enabled (slower but reproducible)")
    else:
        print("Standard reproducibility mode (fast and mostly reproducible)")


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
