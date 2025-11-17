"""
MVSA Dataset Implementation
Supports MVSA-Single and MVSA-Multiple datasets for multimodal sentiment analysis.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, List, Optional, Tuple
import json


class MVSADataset(Dataset):
    """
    MVSA (Multimodal Visual Sentiment Analysis) Dataset.
    Supports both MVSA-Single and MVSA-Multiple variants.
    """

    def __init__(
        self,
        data_file: str,
        image_dir: str,
        text_processor,
        image_processor,
        label_map: Dict[str, int] = None,
        max_text_length: int = 77,
        augment: bool = False
    ):
        """
        Initialize MVSA dataset.

        Args:
            data_file: Path to CSV file with columns: [id, image_path, text, label]
            image_dir: Root directory containing images
            text_processor: Text preprocessing function (e.g., CLIPTextProcessor)
            image_processor: Image preprocessing function (e.g., CLIPImageProcessor)
            label_map: Mapping from label strings to integers
            max_text_length: Maximum text sequence length
            augment: Whether to apply data augmentation
        """
        super().__init__()

        self.data_file = data_file
        self.image_dir = image_dir
        self.text_processor = text_processor
        self.image_processor = image_processor
        self.max_text_length = max_text_length
        self.augment = augment

        # Default label mapping for sentiment analysis
        if label_map is None:
            label_map = {
                'positive': 0,
                'negative': 1,
                'neutral': 2
            }
        self.label_map = label_map
        self.num_classes = len(label_map)

        # Load data
        self.data = self._load_data()

        print(f"Loaded {len(self.data)} samples from {data_file}")
        print(f"Label distribution: {self._get_label_distribution()}")

    def _load_data(self) -> pd.DataFrame:
        """Load data from CSV file."""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        # Load CSV
        df = pd.read_csv(self.data_file)

        # Verify required columns
        required_cols = ['id', 'image_path', 'text', 'label']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Map labels to integers
        df['label_id'] = df['label'].map(self.label_map)

        # Remove samples with invalid labels
        invalid_mask = df['label_id'].isna()
        if invalid_mask.any():
            print(f"Warning: Removing {invalid_mask.sum()} samples with invalid labels")
            df = df[~invalid_mask]

        df['label_id'] = df['label_id'].astype(int)

        return df.reset_index(drop=True)

    def _get_label_distribution(self) -> Dict[str, int]:
        """Get label distribution."""
        return self.data['label'].value_counts().to_dict()

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            dict with:
                - input_ids: [seq_len] text token IDs
                - attention_mask: [seq_len] text attention mask
                - pixel_values: [C, H, W] image pixels
                - label: scalar label ID
                - sample_id: string sample ID
        """
        row = self.data.iloc[idx]

        # Load text
        text = row['text']
        if pd.isna(text):
            text = ""  # Handle missing text

        # Process text
        text_inputs = self.text_processor(text)

        # Load image
        image_path = os.path.join(self.image_dir, row['image_path'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Use blank image as fallback
            image = Image.new('RGB', (224, 224), color='white')

        # Process image
        image_inputs = self.image_processor(image)

        # Get label
        label = torch.tensor(row['label_id'], dtype=torch.long)

        # Combine
        sample = {
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'pixel_values': image_inputs['pixel_values'].squeeze(0),
            'label': label,
            'sample_id': str(row['id'])
        }

        return sample

    def get_label_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced datasets.

        Returns:
            weights: [num_classes] class weights (inverse frequency)
        """
        label_counts = self.data['label_id'].value_counts().sort_index()
        total = len(self.data)
        weights = torch.tensor([
            total / label_counts[i] if i in label_counts.index else 1.0
            for i in range(self.num_classes)
        ], dtype=torch.float)
        return weights


class MVSASingleDataset(MVSADataset):
    """
    MVSA-Single dataset variant.
    Each sample has a single sentiment label for the entire image-text pair.
    """

    def __init__(self, *args, **kwargs):
        # Remove use_soft_labels if present (only applies to MVSA-Multiple)
        kwargs.pop('use_soft_labels', None)

        # Default label mapping for MVSA-Single
        if 'label_map' not in kwargs:
            kwargs['label_map'] = {
                'positive': 0,
                'negative': 1,
                'neutral': 2
            }
        super().__init__(*args, **kwargs)


class MVSAMultipleDataset(MVSADataset):
    """
    MVSA-Multiple dataset variant.
    Samples may have multiple sentiment labels from different annotators.
    This implementation uses majority voting or label distribution.
    """

    def __init__(
        self,
        *args,
        use_soft_labels: bool = False,
        **kwargs
    ):
        """
        Initialize MVSA-Multiple dataset.

        Args:
            use_soft_labels: If True, use label distribution as soft labels
            *args, **kwargs: Other arguments passed to MVSADataset
        """
        self.use_soft_labels = use_soft_labels
        super().__init__(*args, **kwargs)

    def _load_data(self) -> pd.DataFrame:
        """Load data with multiple labels."""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        df = pd.read_csv(self.data_file)

        # MVSA-Multiple may have 'labels' column with list of labels
        if 'labels' in df.columns and 'label' not in df.columns:
            # Parse labels (assuming comma-separated or JSON list)
            df['label_list'] = df['labels'].apply(self._parse_labels)

            if self.use_soft_labels:
                # Compute soft label distribution
                df['label_dist'] = df['label_list'].apply(self._compute_label_distribution)
            else:
                # Use majority vote
                df['label'] = df['label_list'].apply(self._majority_vote)

        # Continue with standard processing
        return super()._load_data()

    def _parse_labels(self, labels_str: str) -> List[str]:
        """Parse labels from string."""
        if pd.isna(labels_str):
            return []
        try:
            # Try JSON parsing
            labels = json.loads(labels_str)
            if isinstance(labels, list):
                return labels
        except:
            pass
        # Try comma-separated
        return [l.strip() for l in str(labels_str).split(',')]

    def _majority_vote(self, label_list: List[str]) -> str:
        """Get majority label from list."""
        if not label_list:
            return 'neutral'  # Default
        from collections import Counter
        counts = Counter(label_list)
        return counts.most_common(1)[0][0]

    def _compute_label_distribution(self, label_list: List[str]) -> torch.Tensor:
        """Compute soft label distribution."""
        dist = torch.zeros(self.num_classes)
        if not label_list:
            # Uniform distribution if no labels
            dist.fill_(1.0 / self.num_classes)
            return dist

        for label in label_list:
            if label in self.label_map:
                dist[self.label_map[label]] += 1

        # Normalize
        total = dist.sum()
        if total > 0:
            dist = dist / total
        else:
            dist.fill_(1.0 / self.num_classes)

        return dist

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get sample with optional soft labels."""
        sample = super().__getitem__(idx)

        if self.use_soft_labels:
            row = self.data.iloc[idx]
            if 'label_dist' in row:
                sample['label_dist'] = row['label_dist']

        return sample


def create_mvsa_dataset(
    dataset_type: str,
    data_file: str,
    image_dir: str,
    text_processor,
    image_processor,
    **kwargs
) -> MVSADataset:
    """
    Factory function to create MVSA dataset.

    Args:
        dataset_type: 'single' or 'multiple'
        data_file: Path to data CSV
        image_dir: Path to image directory
        text_processor: Text preprocessing function
        image_processor: Image preprocessing function
        **kwargs: Additional arguments

    Returns:
        MVSADataset instance
    """
    if dataset_type.lower() == 'single':
        return MVSASingleDataset(
            data_file=data_file,
            image_dir=image_dir,
            text_processor=text_processor,
            image_processor=image_processor,
            **kwargs
        )
    elif dataset_type.lower() == 'multiple':
        return MVSAMultipleDataset(
            data_file=data_file,
            image_dir=image_dir,
            text_processor=text_processor,
            image_processor=image_processor,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for MVSA datasets.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched tensors
    """
    # Stack tensors
    input_ids = torch.stack([sample['input_ids'] for sample in batch])
    attention_mask = torch.stack([sample['attention_mask'] for sample in batch])
    pixel_values = torch.stack([sample['pixel_values'] for sample in batch])
    labels = torch.stack([sample['label'] for sample in batch])

    collated = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'labels': labels,
        'sample_ids': [sample['sample_id'] for sample in batch]
    }

    # Handle soft labels if present
    if 'label_dist' in batch[0]:
        label_dists = torch.stack([sample['label_dist'] for sample in batch])
        collated['label_dists'] = label_dists

    return collated
