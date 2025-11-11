#!/bin/bash
# Training Script for CDAN Model

set -e  # Exit on error

echo "========================================"
echo "Training CDAN Model"
echo "========================================"

# Default configuration
CONFIG="configs/cdan.yaml"
EXPERIMENT_NAME="cdan_mvsa"
EPOCHS=30
BATCH_SIZE=32
LR=0.001
CLIP_LR=0.00001

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --clip-lr)
            CLIP_LR="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Config file: $CONFIG"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LR"
echo "  CLIP learning rate: $CLIP_LR"
if [ ! -z "$RESUME" ]; then
    echo "  Resume from: $RESUME"
fi
echo ""

# Check if data exists
if [ ! -f "data/processed/train.csv" ]; then
    echo "ERROR: Training data not found!"
    echo "Please run: bash scripts/prepare_data.sh"
    exit 1
fi

# Create output directories
mkdir -p checkpoints
mkdir -p logs

# Build command
CMD="python src/train.py \
    --config $CONFIG \
    --experiment-name $EXPERIMENT_NAME \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --clip-lr $CLIP_LR"

if [ ! -z "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

# Run training
echo "Starting training..."
echo "Command: $CMD"
echo ""

$CMD

echo ""
echo "========================================"
echo "Training completed!"
echo "========================================"
echo ""
echo "Check results in:"
echo "  Checkpoints: checkpoints/$EXPERIMENT_NAME_*/"
echo "  Logs: logs/$EXPERIMENT_NAME_*/"
echo ""
