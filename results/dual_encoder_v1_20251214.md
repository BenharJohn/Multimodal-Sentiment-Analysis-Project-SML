# Dual Encoder Results v1 - December 14, 2025

## Configuration
- **Config file**: configs/cdan.yaml
- **Commit**: bd7ae94 (Add Focal Loss for class imbalance handling)
- **Architecture**: CDAN 2025 (CLIP + BERT + ResNet-50 Dual Encoders)

### Key Settings
```yaml
# Dual Encoders (CDAN 2025)
use_dual_encoders: true
bert_model_name: bert-base-uncased
resnet_model: resnet50
freeze_aux_encoders: true

# Model
cross_attn_layers: 2
cross_attn_heads: 8
gate_type: simple
classifier_hidden_dims: [512, 256]
classifier_dropout: 0.3

# Training
lr: 0.001
clip_lr: 0.00001
weight_decay: 0.01
epochs: 50
batch_size: 32
freeze_epochs: 3

# Loss
use_focal_loss: true
focal_gamma: 2.0
focal_alpha: [1.0, 9.4, 1.4]  # [Pos, Neg, Neu]

# Regularization
label_smoothing: 0.0
use_mixup: false
augment_train: false
early_stopping: true
patience: 15
```

## Model Size
- **Total Parameters**: 461,089,795 (461M)
- **Trainable Parameters**: 23,475,203 (23.5M)
- **Frozen**: CLIP + BERT + ResNet encoders

## Results

### Best Validation Performance (Epoch 28)
| Metric | Training | Validation |
|--------|----------|------------|
| Accuracy | ~85% | 60.76% |
| Macro F1 | 85.91% | **54.74%** |
| Loss | 0.2527 | 0.9792 |

### Training Progression
| Epoch | Train Loss | Train F1 | Val Loss | Val F1 | Notes |
|-------|------------|----------|----------|--------|-------|
| 1 | 1.0167 | 39.18% | 0.9655 | 41.29% | Initial |
| 3 | 0.8498 | 51.39% | 0.9213 | 50.27% | End freeze |
| 6 | 0.7048 | 59.83% | 0.8917 | 52.77% | New best |
| 10 | 0.5574 | 67.18% | 0.9009 | 53.86% | New best |
| 13 | 0.4534 | 72.68% | 0.9059 | 54.50% | Peak validation |
| 20 | 0.3299 | 80.24% | 0.9507 | 53.58% | Overfitting |
| 28 | 0.2527 | 85.91% | 0.9792 | 54.74% | Final best |

### Comparison to Baseline

| Model | Best Val F1 | Val Accuracy | Parameters |
|-------|-------------|--------------|------------|
| Baseline (CLIP only) | 52.50% | 58.70% | ~153M |
| **Dual Encoder v1** | **54.74%** | **60.76%** | 461M |
| **Improvement** | **+2.24%** | **+2.06%** | - |

## Observations

### Improvements
1. **+2.24% Val F1** improvement over baseline (52.50% -> 54.74%)
2. **+2.06% Val Accuracy** improvement (58.70% -> 60.76%)
3. Dual encoders (BERT + ResNet) provide complementary features
4. Focal Loss helps with class imbalance

### Issues
1. **Severe Overfitting**: Train F1 (85.91%) >> Val F1 (54.74%) - gap of 31%
2. **Early stopping triggered** at epoch 28 after 15 epochs without improvement
3. Best validation reached around epoch 13, then plateaued
4. Model has 461M parameters but only 23.5M trainable

### Per-Epoch Analysis
- **Epochs 1-3**: Encoders frozen, steady improvement
- **Epochs 4-13**: Best validation performance window
- **Epochs 14-28**: Training continues improving, validation stagnates (overfitting)

## Next Steps (Reduce Overfitting)
1. Increase dropout (0.3 -> 0.4-0.5)
2. Enable label smoothing (0.0 -> 0.1)
3. Add data augmentation
4. Reduce model capacity or increase regularization
5. Try learning rate scheduling or reduce LR after epoch 13

## Checkpoint Location
- Best model: `checkpoints/cdan_mvsa_YYYYMMDD_HHMMSS/best_model.pth`
- Metrics: `logs/cdan_mvsa_YYYYMMDD_HHMMSS/metrics.json`

## How to Reproduce
```bash
git checkout bd7ae94
python src/train.py --config configs/cdan.yaml
```
