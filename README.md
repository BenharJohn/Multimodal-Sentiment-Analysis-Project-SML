# Multimodal Sentiment Analysis with CDAN

Production-ready implementation of a CLIP-based Cross-Domain Attention Network (CDAN) for multimodal sentiment analysis on MVSA-Single and MVSA-Multiple datasets.

## Overview

This repository implements a state-of-the-art multimodal sentiment analysis system that fuses image and text modalities using:

- **CLIP Encoders**: Pretrained vision and text encoders from OpenAI's CLIP
- **Bidirectional Cross-Attention**: Two-layer cross-attention mechanism for text→image and image→text fusion
- **Modality Gating**: Adaptive weighting of modalities based on their relevance
- **Auxiliary Reconstruction**: Regularization via reconstruction of frozen CLIP embeddings
- **Progressive Unfreezing**: Strategic training schedule for CLIP fine-tuning

### Key Features

✅ Modular, production-ready codebase
✅ Support for both MVSA-Single and MVSA-Multiple datasets
✅ Comprehensive metrics and visualizations
✅ Flexible configuration system
✅ Early stopping and checkpointing
✅ Attention visualization (optional)

## Architecture

```
Input: Image + Text
    ↓
[CLIP Encoders]
    ├─ Text Encoder → text_tokens [B, T, D] + text_pooled [B, 512]
    └─ Image Encoder → vision_patches [B, P, D] + vision_pooled [B, 512]
    ↓
[Cross-Attention Fusion]
    ├─ Text→Image Attention (2 layers)
    └─ Image→Text Attention (2 layers)
    ↓
[Modality Gating]
    └─ Adaptive modality weighting
    ↓
[Classifier Head]
    └─ MLP → Sentiment logits [B, 3]

Auxiliary Loss:
    └─ Reconstruct frozen CLIP embeddings (L_recon)

Total Loss = L_CE + α * L_recon (α = 0.05)
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/multimodal-sentiment-analysis.git
cd multimodal-sentiment-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Preparation

### MVSA-Single / MVSA-Multiple

1. Download the MVSA datasets from:
   - [MVSA Official](http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/)
   - [Fair MVSA Splits](https://github.com/dykderrick/Fair_MVSA_Tweet_Comparison)

2. Organize your data:

```
data/
├── raw/
│   └── images/
│       ├── tweet_12345.jpg
│       └── ...
└── processed/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

3. CSV format:

```csv
id,image_path,text,label
12345,tweet_12345.jpg,"This is amazing!",positive
12346,tweet_12346.jpg,"Very disappointed",negative
12347,tweet_12347.jpg,"It's okay",neutral
```

### Prepare Data Script

```bash
# View data format guide
bash scripts/prepare_data.sh

# Create dummy dataset for testing
bash scripts/prepare_data.sh --create-dummy
```

## Training

### Quick Start

```bash
# Train with default configuration
bash scripts/train_cdan.sh

# Train with custom settings
bash scripts/train_cdan.sh \
    --experiment-name my_experiment \
    --epochs 40 \
    --batch-size 64 \
    --lr 0.001
```

### Python API

```python
python src/train.py \
    --config configs/cdan.yaml \
    --experiment-name cdan_mvsa \
    --epochs 30 \
    --batch-size 32 \
    --lr 0.001 \
    --clip-lr 0.00001 \
    --freeze-epochs 3
```

### Training Configuration

Edit `configs/cdan.yaml` to customize:

```yaml
# Model
clip_model_name: openai/clip-vit-base-patch32
num_classes: 3
cross_attn_layers: 2
use_gating: true
use_aux_decoder: true
aux_loss_weight: 0.05

# Training
epochs: 30
batch_size: 32
lr: 0.001
clip_lr: 0.00001
freeze_epochs: 3
```

### Training Schedule

1. **Epochs 1-3**: CLIP encoders frozen, train fusion + classifier
2. **Epochs 4+**: Unfreeze CLIP last blocks, fine-tune end-to-end
3. **Early stopping**: Based on validation macro-F1

## Evaluation

### Quick Start

```bash
# Evaluate best model
bash scripts/eval.sh \
    --checkpoint checkpoints/cdan_mvsa_*/best_model.pth \
    --save-predictions \
    --visualize-attention
```

### Python API

```python
python src/evaluate.py \
    --checkpoint checkpoints/best_model.pth \
    --config configs/cdan.yaml \
    --output-dir results \
    --save-predictions \
    --visualize-attention
```

### Outputs

Results are saved to `results/`:

- `metrics.json`: Accuracy, Precision, Recall, F1, AUC
- `confusion_matrix.png`: Confusion matrix visualization
- `per_class_metrics.png`: Per-class performance chart
- `predictions.csv`: Individual predictions (if `--save-predictions`)
- `error_analysis.json`: Detailed error breakdown

## Project Structure

```
multimodal-sentiment/
├── configs/
│   ├── cdan.yaml                    # Base configuration
│   └── cdan_mvsa_multiple.yaml      # MVSA-Multiple config
├── data/
│   ├── raw/                         # Raw images
│   └── processed/                   # CSV files
├── src/
│   ├── data/
│   │   ├── mvsa.py                  # MVSA dataset implementation
│   │   └── datamodule.py            # Data loading & batching
│   ├── models/
│   │   ├── clip_text.py             # CLIP text encoder wrapper
│   │   ├── clip_image.py            # CLIP image encoder wrapper
│   │   ├── cross_attention.py       # Bidirectional cross-attention
│   │   ├── gating.py                # Modality gating mechanisms
│   │   ├── aux_decoder.py           # Auxiliary reconstruction decoder
│   │   └── classifier.py            # Classification heads
│   ├── systems/
│   │   └── cdan_model.py            # Main CDAN model
│   ├── utils/
│   │   ├── metrics.py               # Evaluation metrics
│   │   ├── seed.py                  # Reproducibility utilities
│   │   └── logging.py               # Experiment logging
│   ├── train.py                     # Training script
│   └── evaluate.py                  # Evaluation script
├── scripts/
│   ├── prepare_data.sh              # Data preparation
│   ├── train_cdan.sh                # Training wrapper
│   └── eval.sh                      # Evaluation wrapper
├── checkpoints/                     # Saved models
├── logs/                            # Training logs
├── results/                         # Evaluation results
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Model Details

### CLIP Encoders

- **Model**: `openai/clip-vit-base-patch32`
- **Text**: 512-dim embeddings, max 77 tokens
- **Image**: 512-dim embeddings, 224×224 input

### Cross-Attention Fusion

- **Layers**: 2 bidirectional blocks
- **Heads**: 8 attention heads per layer
- **Pooling**: Mean pooling for text, CLS token for images

### Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Epochs | 30 | Total training epochs |
| Batch Size | 32 | Samples per batch |
| Learning Rate (Head) | 1e-3 | For fusion & classifier |
| Learning Rate (CLIP) | 1e-5 | For CLIP encoders |
| Weight Decay | 0.01 | L2 regularization |
| Freeze Epochs | 3 | Epochs to freeze CLIP |
| Aux Loss Weight | 0.05 | Weight for reconstruction |
| Gradient Clip | 1.0 | Max gradient norm |

## Baseline Comparisons

This implementation adapts techniques from:

1. **SCRD** (CVPR 2025): Cross-attention fusion patterns
   - [Paper](https://openaccess.thecvf.com/)

2. **CS5242 CLIP MVSA**: CLIP integration for MVSA
   - [GitHub](https://github.com/zxcver/CS5242-CLIP-MVSA)

3. **Fair MVSA**: Standard evaluation protocol
   - [GitHub](https://github.com/dykderrick/Fair_MVSA_Tweet_Comparison)

## Advanced Features

### Soft Labels (MVSA-Multiple)

For datasets with multiple annotations:

```yaml
# configs/cdan_mvsa_multiple.yaml
dataset_type: multiple
use_soft_labels: true
label_smoothing: 0.1
```

### Custom Gating Mechanisms

```python
# In configs/cdan.yaml
gate_type: simple     # 'simple', 'attention', 'hierarchical'
gate_activation: sigmoid  # 'sigmoid', 'softmax', 'tanh'
```

### Resume Training

```bash
python src/train.py \
    --resume checkpoints/cdan_mvsa_*/checkpoint_epoch_10.pth
```

## Troubleshooting

### Out of Memory

- Reduce `batch_size` in config
- Use gradient accumulation
- Enable mixed precision (modify train.py)

### Poor Performance

- Increase `freeze_epochs` (more stable initialization)
- Adjust `aux_loss_weight` (0.01 - 0.1)
- Try different `gate_type` settings
- Ensure data quality and class balance

### Slow Training

- Increase `num_workers` in config
- Use `pin_memory: true`
- Ensure data is on fast storage (SSD)

## Citation

If you use this code, please cite:

```bibtex
@software{cdan_mvsa_2025,
  title={CDAN: Cross-Domain Attention Network for Multimodal Sentiment Analysis},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/multimodal-sentiment-analysis}
}
```

### Related Work

```bibtex
@inproceedings{scrd2025,
  title={Self-supervised Cross-modal Representation Distillation},
  booktitle={CVPR},
  year={2025}
}

@misc{mvsa_dataset,
  title={MVSA: A Multi-View Sentiment Analysis Dataset},
  author={Niu et al.},
  year={2016}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

**Note**: Respect the licenses of the original datasets and pretrained models:
- MVSA datasets: [License](http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/)
- CLIP models: [License](https://github.com/openai/CLIP/blob/main/LICENSE)

## Acknowledgments

- OpenAI for CLIP pretrained models
- MVSA dataset creators
- HuggingFace Transformers library
- Reference implementations from SCRD, CS5242, and Fair MVSA

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## Contact

For questions or issues:
- Open an issue on GitHub
- Email: your.email@example.com

## Changelog

### v1.0.0 (2025-01)
- Initial release
- MVSA-Single and MVSA-Multiple support
- CDAN architecture with CLIP
- Comprehensive evaluation metrics
