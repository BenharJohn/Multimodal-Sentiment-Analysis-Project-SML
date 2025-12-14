"""
Data Module for MVSA Datasets
Handles data loading, preprocessing, and batching for training/evaluation.
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from typing import Optional, Dict, Tuple
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.clip_text import CLIPTextProcessor
from models.clip_image import CLIPImageProcessor
from models.bert_encoder import BERTTextProcessor
from data.mvsa import create_mvsa_dataset, collate_fn


class MVSADataModule:
    """
    Data module for MVSA datasets.
    Handles data loading, splitting, and batch creation.
    """

    def __init__(
        self,
        # Data configuration
        dataset_type: str = 'single',  # 'single' or 'multiple'
        data_dir: str = 'data/processed',
        image_dir: str = 'data/raw/images',

        # Split configuration
        train_file: Optional[str] = None,
        val_file: Optional[str] = None,
        test_file: Optional[str] = None,
        val_split: float = 0.1,
        test_split: float = 0.1,
        random_seed: int = 42,

        # Model configuration
        clip_model_name: str = 'openai/clip-vit-base-patch32',
        max_text_length: int = 77,

        # DataLoader configuration
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,

        # Other
        use_soft_labels: bool = False,
        augment_train: bool = False,

        # Dual encoder configuration (CDAN 2025)
        use_dual_encoders: bool = False,
        bert_model_name: str = 'bert-base-uncased',
        bert_max_length: int = 128
    ):
        """
        Initialize data module.

        Args:
            dataset_type: 'single' or 'multiple'
            data_dir: Directory containing data CSV files
            image_dir: Directory containing images
            train_file: Path to training CSV (if None, will look for train.csv)
            val_file: Path to validation CSV (if None, will split from train)
            test_file: Path to test CSV (if None, will split from train)
            val_split: Validation split ratio (if val_file not provided)
            test_split: Test split ratio (if test_file not provided)
            random_seed: Random seed for reproducibility
            clip_model_name: CLIP model identifier
            max_text_length: Maximum text sequence length
            batch_size: Batch size for dataloaders
            num_workers: Number of worker processes for dataloading
            pin_memory: Whether to pin memory for faster GPU transfer
            use_soft_labels: Whether to use soft labels (for MVSA-Multiple)
            augment_train: Whether to augment training data
            use_dual_encoders: Whether to use BERT + CLIP dual encoders (CDAN 2025)
            bert_model_name: BERT model name for auxiliary text encoder
            bert_max_length: Maximum sequence length for BERT tokenizer
        """
        self.dataset_type = dataset_type
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.random_seed = random_seed
        self.use_soft_labels = use_soft_labels
        self.augment_train = augment_train
        self.use_dual_encoders = use_dual_encoders

        # File paths
        self.train_file = train_file or os.path.join(data_dir, 'train.csv')
        self.val_file = val_file
        self.test_file = test_file
        self.val_split = val_split
        self.test_split = test_split

        # Initialize CLIP processors
        self.text_processor = CLIPTextProcessor(
            model_name=clip_model_name,
            max_length=max_text_length
        )
        self.image_processor = CLIPImageProcessor(
            model_name=clip_model_name
        )

        # Initialize BERT processor for dual encoders (CDAN 2025)
        self.bert_processor = None
        if use_dual_encoders:
            self.bert_processor = BERTTextProcessor(
                model_name=bert_model_name,
                max_length=bert_max_length
            )

        # Datasets (initialized in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Label information
        self.num_classes = 3  # Default for sentiment
        self.label_map = {
            'positive': 0,
            'negative': 1,
            'neutral': 2
        }

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training/validation/testing.

        Args:
            stage: 'fit', 'validate', 'test', or None (setup all)
        """
        # Training data
        if stage in ['fit', None] and self.train_dataset is None:
            if os.path.exists(self.train_file):
                # Load from separate train file
                if self.val_file and os.path.exists(self.val_file):
                    # Load train and val separately
                    self.train_dataset = self._create_dataset(
                        self.train_file,
                        augment=self.augment_train
                    )
                    self.val_dataset = self._create_dataset(
                        self.val_file,
                        augment=False
                    )
                else:
                    # Load train and split for validation
                    full_dataset = self._create_dataset(
                        self.train_file,
                        augment=False  # Will apply after split
                    )
                    self._split_dataset(full_dataset)
            else:
                raise FileNotFoundError(f"Training file not found: {self.train_file}")

        # Test data
        if stage in ['test', None] and self.test_dataset is None:
            if self.test_file and os.path.exists(self.test_file):
                self.test_dataset = self._create_dataset(
                    self.test_file,
                    augment=False
                )
            elif stage == 'test':
                print("Warning: No test file provided. Using validation set for testing.")
                if self.val_dataset is None:
                    self.setup('fit')
                self.test_dataset = self.val_dataset

        # Print dataset info
        if self.train_dataset:
            print(f"Training samples: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"Validation samples: {len(self.val_dataset)}")
        if self.test_dataset:
            print(f"Test samples: {len(self.test_dataset)}")

    def _create_dataset(self, data_file: str, augment: bool = False):
        """Create a dataset instance."""
        return create_mvsa_dataset(
            dataset_type=self.dataset_type,
            data_file=data_file,
            image_dir=self.image_dir,
            text_processor=self.text_processor,
            image_processor=self.image_processor,
            label_map=self.label_map,
            augment=augment,
            use_soft_labels=self.use_soft_labels,
            bert_processor=self.bert_processor,
            use_dual_encoders=self.use_dual_encoders
        )

    def _split_dataset(self, full_dataset):
        """Split dataset into train/val/test."""
        total_size = len(full_dataset)

        # Calculate splits
        if self.test_split > 0:
            test_size = int(total_size * self.test_split)
        else:
            test_size = 0

        if self.val_split > 0:
            val_size = int(total_size * self.val_split)
        else:
            val_size = 0

        train_size = total_size - val_size - test_size

        # Perform split
        generator = torch.Generator().manual_seed(self.random_seed)

        if test_size > 0:
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full_dataset,
                [train_size, val_size, test_size],
                generator=generator
            )
        else:
            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=generator
            )

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        if self.train_dataset is None:
            self.setup('fit')

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            drop_last=True  # Drop last incomplete batch for stability
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        if self.val_dataset is None:
            self.setup('fit')

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        if self.test_dataset is None:
            self.setup('test')

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )

    def get_class_weights(self) -> Optional[torch.Tensor]:
        """Get class weights for handling imbalanced data."""
        if self.train_dataset is None:
            self.setup('fit')

        # If train_dataset is a Subset, access the underlying dataset
        dataset = self.train_dataset
        if hasattr(dataset, 'dataset'):
            dataset = dataset.dataset

        if hasattr(dataset, 'get_label_weights'):
            return dataset.get_label_weights()

        return None


def build_datamodule(config: Dict) -> MVSADataModule:
    """
    Build data module from configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        Initialized MVSADataModule
    """
    return MVSADataModule(
        dataset_type=config.get('dataset_type', 'single'),
        data_dir=config.get('data_dir', 'data/processed'),
        image_dir=config.get('image_dir', 'data/raw/images'),
        train_file=config.get('train_file'),
        val_file=config.get('val_file'),
        test_file=config.get('test_file'),
        val_split=config.get('val_split', 0.1),
        test_split=config.get('test_split', 0.1),
        random_seed=config.get('random_seed', 42),
        clip_model_name=config.get('clip_model', config.get('clip_model_name', 'openai/clip-vit-base-patch32')),
        max_text_length=config.get('max_text_length', 77),
        batch_size=config.get('batch_size', 32),
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True),
        use_soft_labels=config.get('use_soft_labels', False),
        augment_train=config.get('augment_train', False),
        # Dual encoder configuration (CDAN 2025)
        use_dual_encoders=config.get('use_dual_encoders', False),
        bert_model_name=config.get('bert_model_name', 'bert-base-uncased'),
        bert_max_length=config.get('bert_max_length', 128)
    )
