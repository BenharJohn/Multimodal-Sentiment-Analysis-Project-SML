#!/usr/bin/env python3
"""
Convert MVSA-Multiple dataset to our training format.

The MVSA-Multiple dataset has:
- data/{ID}.jpg - Tweet image
- data/{ID}.txt - Tweet text
- labelResultAll.txt - Multi-annotator labels (3 annotators per sample)

Each annotator provides: (text_sentiment, image_sentiment)
We use majority vote for final label and also save soft label distributions.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import argparse
import shutil
from tqdm import tqdm


def parse_label_line(line):
    """
    Parse a label line from labelResultAll.txt

    Format: ID    text,image    text,image    text,image
    Example: 2499    positive,positive    neutral,neutral    positive,positive

    Returns: (id, [text_labels], [image_labels])
    """
    parts = line.strip().split('\t')
    if len(parts) < 4:
        return None

    sample_id = parts[0].strip()

    text_labels = []
    image_labels = []

    for i in range(1, 4):  # 3 annotators
        if i >= len(parts):
            continue
        annotation = parts[i].strip()
        if ',' in annotation:
            text_label, image_label = annotation.split(',')
            text_labels.append(text_label.strip())
            image_labels.append(image_label.strip())

    return sample_id, text_labels, image_labels


def majority_vote(labels):
    """Get majority vote from list of labels."""
    if not labels:
        return None
    counter = Counter(labels)
    most_common = counter.most_common(1)[0]
    return most_common[0]


def get_label_distribution(labels):
    """
    Get soft label distribution.
    Returns dict: {'positive': 0.33, 'negative': 0.33, 'neutral': 0.33}
    """
    if not labels:
        return None

    counter = Counter(labels)
    total = len(labels)

    # Ensure all three classes are present
    dist = {
        'positive': counter.get('positive', 0) / total,
        'negative': counter.get('negative', 0) / total,
        'neutral': counter.get('neutral', 0) / total
    }

    return dist


def load_mvsa_labels(label_file):
    """
    Load all labels from labelResultAll.txt

    Returns: DataFrame with columns: id, text_label, image_label,
             text_dist, image_dist, agreement_score
    """
    samples = []

    with open(label_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Skip header
    for line in tqdm(lines[1:], desc="Loading labels"):
        result = parse_label_line(line)
        if result is None:
            continue

        sample_id, text_labels, image_labels = result

        # Get majority vote
        text_label = majority_vote(text_labels)
        image_label = majority_vote(image_labels)

        # Get label distributions
        text_dist = get_label_distribution(text_labels)
        image_dist = get_label_distribution(image_labels)

        # Calculate agreement score (how many annotators agreed)
        text_agreement = Counter(text_labels).most_common(1)[0][1] / len(text_labels)
        image_agreement = Counter(image_labels).most_common(1)[0][1] / len(image_labels)

        # For multimodal sentiment, use text label as primary
        # (you can also use image label or combine them)
        final_label = text_label

        samples.append({
            'id': sample_id,
            'text_label': text_label,
            'image_label': image_label,
            'label': final_label,  # Primary label for classification
            'text_dist_positive': text_dist['positive'],
            'text_dist_negative': text_dist['negative'],
            'text_dist_neutral': text_dist['neutral'],
            'image_dist_positive': image_dist['positive'],
            'image_dist_negative': image_dist['negative'],
            'image_dist_neutral': image_dist['neutral'],
            'text_agreement': text_agreement,
            'image_agreement': image_agreement,
            'num_annotators': len(text_labels)
        })

    return pd.DataFrame(samples)


def load_sample_data(data_dir, sample_id):
    """
    Load text and check if image exists for a sample.

    Returns: (text, image_path) or (None, None) if missing
    """
    text_file = os.path.join(data_dir, f"{sample_id}.txt")
    image_file = os.path.join(data_dir, f"{sample_id}.jpg")

    # Check if both files exist
    if not os.path.exists(text_file) or not os.path.exists(image_file):
        return None, None

    # Read text
    try:
        with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read().strip()
    except Exception as e:
        print(f"Error reading {text_file}: {e}")
        return None, None

    return text, image_file


def create_dataset(mvsa_dir, output_dir, split_ratios=(0.7, 0.15, 0.15),
                   min_agreement=0.5, seed=42):
    """
    Create train/val/test datasets from MVSA-Multiple.

    Args:
        mvsa_dir: Path to MVSA directory (contains data/ and labelResultAll.txt)
        output_dir: Output directory for processed data
        split_ratios: (train, val, test) ratios
        min_agreement: Minimum annotator agreement (0.33-1.0)
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    # Paths
    data_dir = os.path.join(mvsa_dir, 'data')
    label_file = os.path.join(mvsa_dir, 'labelResultAll.txt')

    # Create output directories
    processed_dir = os.path.join(output_dir, 'processed')
    image_dir = os.path.join(output_dir, 'raw', 'images')
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    print("="*60)
    print("MVSA-Multiple Dataset Preparation")
    print("="*60)
    print(f"Source: {mvsa_dir}")
    print(f"Output: {output_dir}")
    print(f"Min agreement: {min_agreement}")
    print()

    # Load labels
    print("Step 1: Loading labels...")
    labels_df = load_mvsa_labels(label_file)
    print(f"  Total samples with labels: {len(labels_df)}")

    # Filter by agreement
    if min_agreement > 0.33:
        print(f"\nStep 2: Filtering by agreement >= {min_agreement}...")
        labels_df = labels_df[labels_df['text_agreement'] >= min_agreement]
        print(f"  Samples after filtering: {len(labels_df)}")

    # Load data and create dataset
    print("\nStep 3: Loading text and verifying images...")
    dataset = []
    missing_count = 0

    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Processing"):
        sample_id = row['id']
        text, image_path = load_sample_data(data_dir, sample_id)

        if text is None or image_path is None:
            missing_count += 1
            continue

        # Copy image to output directory with consistent naming
        new_image_name = f"{sample_id}.jpg"
        new_image_path = os.path.join(image_dir, new_image_name)

        try:
            shutil.copy2(image_path, new_image_path)
        except Exception as e:
            print(f"Error copying image {sample_id}: {e}")
            continue

        dataset.append({
            'id': sample_id,
            'image_path': new_image_name,
            'text': text,
            'label': row['label'],
            'text_label': row['text_label'],
            'image_label': row['image_label'],
            'text_agreement': row['text_agreement'],
            'image_agreement': row['image_agreement']
        })

    print(f"  Successfully loaded: {len(dataset)} samples")
    print(f"  Missing files: {missing_count} samples")

    # Convert to DataFrame
    df = pd.DataFrame(dataset)

    # Print label distribution
    print("\nLabel Distribution:")
    print(df['label'].value_counts())
    print(f"\nText-Image Agreement:")
    print(f"  Same label: {(df['text_label'] == df['image_label']).sum()} "
          f"({100 * (df['text_label'] == df['image_label']).mean():.1f}%)")

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split dataset
    print(f"\nStep 4: Splitting dataset {split_ratios}...")
    n = len(df)
    train_size = int(n * split_ratios[0])
    val_size = int(n * split_ratios[1])

    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]

    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")

    # Save CSV files (with minimal columns for training)
    print("\nStep 5: Saving CSV files...")
    train_cols = ['id', 'image_path', 'text', 'label']
    train_df[train_cols].to_csv(os.path.join(processed_dir, 'train.csv'), index=False)
    val_df[train_cols].to_csv(os.path.join(processed_dir, 'val.csv'), index=False)
    test_df[train_cols].to_csv(os.path.join(processed_dir, 'test.csv'), index=False)

    # Save full metadata (with agreement scores, etc.)
    train_df.to_csv(os.path.join(processed_dir, 'train_metadata.csv'), index=False)
    val_df.to_csv(os.path.join(processed_dir, 'val_metadata.csv'), index=False)
    test_df.to_csv(os.path.join(processed_dir, 'test_metadata.csv'), index=False)

    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print("="*60)
    print(f"\nOutput structure:")
    print(f"  {processed_dir}/train.csv")
    print(f"  {processed_dir}/val.csv")
    print(f"  {processed_dir}/test.csv")
    print(f"  {image_dir}/ ({len(dataset)} images)")
    print()

    # Print sample
    print("Sample data:")
    print(train_df[['id', 'image_path', 'text', 'label']].head(3))
    print()

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Prepare MVSA-Multiple dataset')
    parser.add_argument('--mvsa-dir', type=str, required=True,
                        help='Path to MVSA directory (contains data/ and labelResultAll.txt)')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Output directory (default: data)')
    parser.add_argument('--min-agreement', type=float, default=0.5,
                        help='Minimum annotator agreement (default: 0.5)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Calculate test ratio
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    split_ratios = (args.train_ratio, args.val_ratio, test_ratio)

    # Create dataset
    create_dataset(
        mvsa_dir=args.mvsa_dir,
        output_dir=args.output_dir,
        split_ratios=split_ratios,
        min_agreement=args.min_agreement,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
