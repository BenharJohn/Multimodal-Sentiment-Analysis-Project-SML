#!/bin/bash
# Data Preparation Script for MVSA Datasets
# Downloads and prepares MVSA-Single and MVSA-Multiple datasets

set -e  # Exit on error

echo "========================================"
echo "MVSA Dataset Preparation"
echo "========================================"

# Configuration
DATA_DIR="data"
RAW_DIR="$DATA_DIR/raw"
PROCESSED_DIR="$DATA_DIR/processed"
IMAGE_DIR="$RAW_DIR/images"

# Create directories
echo "Creating directories..."
mkdir -p "$RAW_DIR"
mkdir -p "$PROCESSED_DIR"
mkdir -p "$IMAGE_DIR"

echo ""
echo "NOTE: This script provides a template for data preparation."
echo "You need to:"
echo "  1. Download MVSA-Single/Multiple datasets from the original sources"
echo "  2. Place images in: $IMAGE_DIR"
echo "  3. Prepare CSV files with columns: [id, image_path, text, label]"
echo "  4. Save train/val/test splits in: $PROCESSED_DIR"
echo ""

# Example data format
cat > "$PROCESSED_DIR/data_format_example.txt" << 'EOF'
MVSA Dataset CSV Format:
========================

Required columns:
- id: Unique sample identifier (e.g., tweet ID)
- image_path: Relative path to image from IMAGE_DIR (e.g., "tweet_12345.jpg")
- text: Text content (e.g., tweet text)
- label: Sentiment label ("positive", "negative", or "neutral")

Example CSV:
id,image_path,text,label
12345,tweet_12345.jpg,"This is a great day!",positive
12346,tweet_12346.jpg,"I'm feeling sad",negative
12347,tweet_12347.jpg,"Just a regular day",neutral

File structure:
data/
├── raw/
│   └── images/
│       ├── tweet_12345.jpg
│       ├── tweet_12346.jpg
│       └── tweet_12347.jpg
└── processed/
    ├── train.csv
    ├── val.csv
    └── test.csv

For MVSA-Multiple (multiple annotations):
You can use a "labels" column with comma-separated or JSON list of labels:
id,image_path,text,labels
12345,tweet_12345.jpg,"This is ambiguous","[positive, neutral]"

References:
- MVSA-Single: http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/
- MVSA-Multiple: http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/
- Fair MVSA splits: https://github.com/dykderrick/Fair_MVSA_Tweet_Comparison

Note: Due to Twitter's Terms of Service, tweet IDs are usually shared rather than
full content. You may need to hydrate the dataset using Twitter API or use
pre-processed versions if available.
EOF

echo "Data format guide created at: $PROCESSED_DIR/data_format_example.txt"

# Create a dummy sample dataset for testing (if requested)
if [ "$1" == "--create-dummy" ]; then
    echo ""
    echo "Creating dummy dataset for testing..."

    # Create dummy images
    python3 << 'PYTHON_SCRIPT'
import os
from PIL import Image, ImageDraw, ImageFont
import random

image_dir = "data/raw/images"
os.makedirs(image_dir, exist_ok=True)

# Create 100 dummy images
colors = {
    'positive': (100, 255, 100),
    'negative': (255, 100, 100),
    'neutral': (200, 200, 200)
}

for i in range(100):
    img = Image.new('RGB', (224, 224), color=random.choice(list(colors.values())))
    draw = ImageDraw.Draw(img)
    draw.text((50, 100), f"Sample {i}", fill=(0, 0, 0))
    img.save(os.path.join(image_dir, f"sample_{i:03d}.jpg"))

print("Created 100 dummy images")
PYTHON_SCRIPT

    # Create dummy CSV files
    python3 << 'PYTHON_SCRIPT'
import pandas as pd
import random

texts = [
    "This is amazing!", "Great experience", "Love it!",
    "Terrible service", "Very disappointed", "Not happy",
    "It's okay", "Nothing special", "Average"
]

labels = ['positive', 'negative', 'neutral']

# Generate samples
samples = []
for i in range(100):
    samples.append({
        'id': i,
        'image_path': f"sample_{i:03d}.jpg",
        'text': random.choice(texts),
        'label': random.choice(labels)
    })

df = pd.DataFrame(samples)

# Split: 70% train, 15% val, 15% test
train_df = df[:70]
val_df = df[70:85]
test_df = df[85:]

# Save
train_df.to_csv('data/processed/train.csv', index=False)
val_df.to_csv('data/processed/val.csv', index=False)
test_df.to_csv('data/processed/test.csv', index=False)

print(f"Created dummy dataset:")
print(f"  Train: {len(train_df)} samples")
print(f"  Val: {len(val_df)} samples")
print(f"  Test: {len(test_df)} samples")
PYTHON_SCRIPT

    echo ""
    echo "Dummy dataset created successfully!"
    echo "You can now test the training pipeline."
fi

echo ""
echo "========================================"
echo "Data preparation complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Ensure your data is in the correct format"
echo "  2. Verify file paths and labels"
echo "  3. Run training: bash scripts/train_cdan.sh"
echo ""
