#!/bin/bash
# Evaluation Script for CDAN Model

set -e  # Exit on error

echo "========================================"
echo "Evaluating CDAN Model"
echo "========================================"

# Default configuration
CHECKPOINT=""
CONFIG="configs/cdan.yaml"
OUTPUT_DIR="results"
BATCH_SIZE=32
SAVE_PREDICTIONS=""
VISUALIZE_ATTENTION=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --save-predictions)
            SAVE_PREDICTIONS="--save-predictions"
            shift
            ;;
        --visualize-attention)
            VISUALIZE_ATTENTION="--visualize-attention"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if checkpoint is provided
if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: --checkpoint is required!"
    echo ""
    echo "Usage:"
    echo "  bash scripts/eval.sh --checkpoint <path_to_checkpoint> [options]"
    echo ""
    echo "Options:"
    echo "  --checkpoint <path>       Path to model checkpoint (required)"
    echo "  --config <path>           Path to config file (default: configs/cdan.yaml)"
    echo "  --output-dir <path>       Output directory (default: results)"
    echo "  --batch-size <int>        Batch size (default: 32)"
    echo "  --save-predictions        Save predictions to CSV"
    echo "  --visualize-attention     Generate attention visualizations"
    echo ""
    echo "Example:"
    echo "  bash scripts/eval.sh --checkpoint checkpoints/best_model.pth --save-predictions"
    exit 1
fi

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

# Check if test data exists
if [ ! -f "data/processed/test.csv" ]; then
    echo "WARNING: Test data not found at data/processed/test.csv"
    echo "Using validation set for evaluation..."
fi

echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Config file: $CONFIG"
echo "  Output directory: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Save predictions: ${SAVE_PREDICTIONS:-No}"
echo "  Visualize attention: ${VISUALIZE_ATTENTION:-No}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python src/evaluate.py \
    --checkpoint $CHECKPOINT \
    --config $CONFIG \
    --output-dir $OUTPUT_DIR \
    --batch-size $BATCH_SIZE \
    $SAVE_PREDICTIONS \
    $VISUALIZE_ATTENTION"

# Run evaluation
echo "Starting evaluation..."
echo "Command: $CMD"
echo ""

$CMD

echo ""
echo "========================================"
echo "Evaluation completed!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "  - metrics.json: Detailed metrics"
echo "  - confusion_matrix.png: Confusion matrix visualization"
echo "  - per_class_metrics.png: Per-class performance"
if [ ! -z "$SAVE_PREDICTIONS" ]; then
    echo "  - predictions.csv: Individual predictions"
    echo "  - error_analysis.json: Error analysis"
fi
echo ""
