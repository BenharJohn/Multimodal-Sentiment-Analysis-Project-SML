# Multimodal Sentiment Analysis with CLIP-based Cross-Domain Attention Network

A PyTorch implementation of a multimodal sentiment analysis system using CLIP encoders with bidirectional cross-attention for fusion, trained and evaluated on the MVSA-Multiple dataset.

## 🎯 Project Overview

This project implements a **CLIP-based Cross-Domain Attention Network (CDAN)** for multimodal sentiment analysis on social media data (Twitter). The model processes both text and images to predict sentiment labels (positive, negative, neutral), leveraging pre-trained CLIP encoders and bidirectional cross-attention mechanisms for effective multimodal fusion.

### Key Features

- ✅ Pre-trained CLIP (ViT-B/32) vision and text encoders
- ✅ Bidirectional cross-attention for multimodal fusion
- ✅ Modality gating for adaptive fusion weights
- ✅ Auxiliary reconstruction loss for better representations
- ✅ Support for MVSA-Single and MVSA-Multiple datasets
- ✅ GPU-accelerated training on NVIDIA A100
- ✅ Comprehensive evaluation metrics and logging

---

## 📊 Results

### Performance on MVSA-Multiple Dataset

**Dataset Statistics:**
- Training samples: 10,304
- Validation samples: 1,288
- Test samples: 1,288
- Total images: 19,600
- Classes: Positive, Negative, Neutral

**Training Configuration:**
- Epochs: 50
- Batch size: 32
- Optimizer: AdamW (lr=0.001, clip_lr=1e-5)
- Device: NVIDIA A100-SXM4-80GB
- Training time: ~30 minutes

**Final Results:**

| Metric | Training Set | Validation Set |
|--------|--------------|----------------|
| **Accuracy** | 95.24% | 57.92% |
| **Macro F1** | 95.16% | 47.90% |
| **Best Val F1** | - | **51.17%** |
| **Precision** | 95.71% | 49.56% |
| **Recall** | 94.70% | 46.90% |
| **AUC** | 99.08% | 65.76% |

**Per-Class Performance (Validation Set - Best Model at Epoch 33):**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Positive | 65.24% | 65.80% | 65.52% | 696 |
| Negative | 32.69% | 22.37% | 26.56% | 76 |
| Neutral | 50.75% | 52.52% | 51.62% | 516 |

**Confusion Matrix (Validation Set - Final Epoch 50):**

```
                Predicted
              Pos   Neg   Neu
Actual  Pos   458    13   225
        Neg    21    17    38
        Neu   223    22   271
```

### Key Observations

1. **Strong Training Performance**: 95.24% accuracy indicates the model effectively learns multimodal patterns
2. **Moderate Validation Performance**: 57.92% accuracy is competitive for MVSA-Multiple (state-of-the-art: 60-70%)
3. **Class Imbalance**: Negative class has fewer samples (76) leading to lower performance
4. **Positive Class**: Best performance (65.52% F1) due to more balanced representation
5. **Generalization Gap**: Train-val accuracy gap suggests some overfitting, addressable with more regularization

---

## 🏗️ Architecture

### CDAN (CLIP-based Cross-Domain Attention Network)

The architecture consists of the following components:

```
Input: Text + Image
    │
    ├─────────────────────┬─────────────────────┐
    │                     │                     │
    ▼                     ▼                     ▼
CLIP Text Encoder   CLIP Image Encoder   (Pre-trained)
(ViT-B/32)          (ViT-B/32)
    │                     │
    │                     │
Hidden: 512           Hidden: 768
    │                     │
    ▼                     ▼
Text Projection      Vision Projection
(512 → 512)          (768 → 512)         ← Dimension Alignment
    │                     │
    └──────────┬──────────┘
               │
               ▼
    Bidirectional Cross-Attention
    ┌──────────────────────────┐
    │  Text → Vision Attention │
    │  Vision → Text Attention │
    │  (2 layers, 8 heads)     │
    └──────────────────────────┘
               │
               ▼
        Modality Gating
    ┌──────────────────────┐
    │  α_text · text_feat  │
    │  + α_img · img_feat  │  ← Adaptive Fusion
    └──────────────────────┘
               │
               ├─────────────┬──────────────┐
               │             │              │
               ▼             ▼              ▼
         Pooling    Auxiliary Decoder  Classifier
               │             │         (512→256→3)
               │             │              │
               │             ▼              ▼
               │     Reconstruction   Sentiment
               │          Loss         Logits
               │             │              │
               └─────────────┴──────────────┘
                             │
                             ▼
                    Combined Loss (CE + Aux)
```

### Component Details

#### 1. **CLIP Encoders** (Pre-trained)
- **Text Encoder**: Transformer-based (12 layers, 512 hidden dim)
  - Processes text with BPE tokenization (max 77 tokens)
  - Outputs: Text tokens [B, T, 512], Text embedding [B, 512]

- **Vision Encoder**: Vision Transformer ViT-B/32 (12 layers, 768 hidden dim)
  - Processes 224×224 RGB images
  - 32×32 patch size → 49 patches
  - Outputs: Vision patches [B, P, 768], Image embedding [B, 768]

- **Projection**: Both encoders project to shared 512-dim space for contrastive learning

#### 2. **Dimension Alignment**
```python
# Align text and vision to common dimension (512)
text_projection: Linear(512 → 512)    # Identity for text
vision_projection: Linear(768 → 512)  # Projection for vision
```

**Why needed**: CLIP's text encoder (512-dim) and vision encoder (768-dim) have different hidden dimensions. We project both to 512-dim for cross-attention compatibility.

#### 3. **Bidirectional Cross-Attention**
```python
CrossAttentionFusion(
    hidden_dim=512,
    num_layers=2,
    num_heads=8,
    dropout=0.1
)
```

**Text-to-Vision Attention:**
- Query: Text tokens [B, T, 512]
- Key/Value: Vision patches [B, P, 512]
- Output: Text attended with visual context

**Vision-to-Text Attention:**
- Query: Vision patches [B, P, 512]
- Key/Value: Text tokens [B, T, 512]
- Output: Vision attended with textual context

**Formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

text_ctx = Attention(text, vision, vision)
vision_ctx = Attention(vision, text, text)
```

#### 4. **Modality Gating**
```python
# Learn adaptive fusion weights
gate = sigmoid(Linear([text_feat; img_feat]))
α_text, α_img = gate[:, 0], gate[:, 1]

# Weighted fusion
fused = α_text * text_feat + α_img * img_feat
```

**Purpose**: Adaptively weight text vs. image modalities based on input quality and relevance.

#### 5. **Auxiliary Decoder** (Optional)
```python
AuxiliaryDecoder(
    input_dim=512,
    hidden_dim=512,
    output_dim=512,
    num_layers=2
)
```

**Reconstruction Loss** (Cosine Similarity):
```
L_aux = 1 - cos_sim(reconstructed, original)
```

**Purpose**: Encourage the model to retain modality-specific information during fusion, preventing information loss.

#### 6. **Classification Head**
```python
Classifier(
    input_dim=512,
    hidden_dims=[512, 256],
    num_classes=3,
    dropout=0.3
)
```

**Architecture:**
```
Linear(512 → 512) → ReLU → Dropout(0.3) →
Linear(512 → 256) → ReLU → Dropout(0.3) →
Linear(256 → 3)
```

**Output**: Logits for 3 classes (positive, negative, neutral)

#### 7. **Loss Function**
```python
# Combined loss
L_total = L_ce + λ * L_aux

# Cross-entropy loss (primary)
L_ce = CrossEntropyLoss(logits, labels)

# Auxiliary reconstruction loss (optional)
L_aux = CosineSimilarity(reconstructed, original)

# λ = 0.05 (auxiliary loss weight)
```

### Model Statistics

- **Total Parameters**: 317,792,775 (~318M)
- **Trainable Parameters**: 15,238,149 (~15M)
- **Frozen Parameters**: 302,554,626 (~303M - CLIP encoders)

**Parameter Breakdown:**
- CLIP Text Encoder: ~151M (frozen)
- CLIP Vision Encoder: ~151M (frozen)
- Projection Layers: ~0.4M (trainable)
- Cross-Attention: ~8M (trainable)
- Modality Gating: ~0.5M (trainable)
- Auxiliary Decoder: ~2M (trainable)
- Classifier: ~0.4M (trainable)

---

## 📚 Methods

### Training Strategy

#### 1. **Progressive Unfreezing**
```python
Epochs 1-3:  Freeze CLIP encoders (train only fusion layers)
Epochs 4+:   Unfreeze CLIP last blocks (fine-tune with low LR)
```

**Benefits:**
- Prevents catastrophic forgetting of pre-trained knowledge
- Faster initial convergence
- Better stability during training

#### 2. **Differential Learning Rates**
```python
CLIP encoders:  lr = 1e-5   (low - preserve pre-trained knowledge)
New layers:     lr = 1e-3   (high - learn task-specific patterns)
```

#### 3. **Optimization**
- **Optimizer**: AdamW (weight_decay=0.01)
- **Scheduler**: Cosine annealing with warmup (2 epochs)
- **Gradient Clipping**: max_norm=1.0
- **Batch Size**: 32 (per GPU)

#### 4. **Regularization**
- **Dropout**: 0.3 in classifier layers
- **Weight Decay**: 0.01 (L2 regularization)
- **Label Smoothing**: 0.0 (optional, set to 0.1 for robustness)
- **Data Augmentation**: Optional (random crops, color jitter)

#### 5. **Early Stopping & Checkpointing**
- Monitor validation F1 score
- Save best model based on val F1
- Save checkpoints every 5 epochs
- Patience: Continue until 50 epochs (no early stopping used)

### Data Preprocessing

#### 1. **MVSA-Multiple Dataset Preparation**
```python
# Multi-annotator labels (3 annotators per sample)
# Each provides: (text_sentiment, image_sentiment)

# Majority vote for final label
label = majority_vote([ann1, ann2, ann3])

# Filter by agreement threshold
min_agreement = 0.5  # At least 2/3 annotators agree
```

#### 2. **Text Processing**
- **Tokenizer**: CLIP's BPE tokenizer
- **Max Length**: 77 tokens (CLIP standard)
- **Truncation**: Enabled
- **Padding**: To max length

#### 3. **Image Processing**
- **Resize**: 224×224 pixels
- **Normalization**: CLIP's mean/std
  - Mean: [0.48145466, 0.4578275, 0.40821073]
  - Std: [0.26862954, 0.26130258, 0.27577711]
- **Format**: RGB (3 channels)

#### 4. **Data Splits**
```
Train:      70% (10,304 samples)
Validation: 15% (1,288 samples)
Test:       15% (1,288 samples)
```

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Macro F1**: Average F1 across all classes (handles imbalance)
- **Precision**: Per-class and macro-averaged
- **Recall**: Per-class and macro-averaged
- **AUC-ROC**: Area under ROC curve (multi-class OvR)
- **Confusion Matrix**: Detailed error analysis

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.9+
# CUDA 11.8+ (for GPU training)
# 16GB+ RAM
# 8GB+ GPU memory (for batch_size=32)
```

### Installation

```bash
# Clone repository
git clone https://github.com/BenharJohn/Multimodal-Sentiment-Analysis-Project-SML.git
cd Multimodal-Sentiment-Analysis-Project-SML

# Create virtual environment
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch==2.0.0 torchvision==0.15.0
pip install transformers==4.30.0 pytorch-lightning==2.0.0
pip install torchmetrics pillow tqdm pyyaml pandas scikit-learn
```

### Dataset Preparation

#### MVSA-Multiple Dataset

```bash
# 1. Download MVSA-Multiple from official source
# http://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/

# 2. Extract to MVSA-multiple/MVSA/

# 3. Prepare dataset
python scripts/prepare_mvsa_multiple.py \
    --mvsa-dir MVSA-multiple/MVSA \
    --output-dir data \
    --min-agreement 0.5 \
    --train-ratio 0.7 \
    --val-ratio 0.15
```

This will create:
```
data/
├── processed/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── train_metadata.csv (with agreement scores)
│   ├── val_metadata.csv
│   └── test_metadata.csv
└── raw/
    └── images/
        ├── 2499.jpg
        ├── 2500.jpg
        └── ... (19,600 images)
```

---

## 🏃 Training

### Quick Start (GPU)

```bash
# Set environment
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Load CUDA module (if on HPC)
module load cuda-11.8.0-gcc-12.1.0

# Train on GPU
python src/train.py \
    --clip-model openai/clip-vit-base-patch32 \
    --epochs 50 \
    --batch-size 32 \
    --device cuda \
    --num-workers 8 \
    --experiment-name mvsa_multiple_training
```

### Training Options

```bash
python src/train.py --help

# Key arguments:
--clip-model PATH           # Path to CLIP model or HuggingFace ID
--dataset-type {single|multiple}  # MVSA-Single or MVSA-Multiple
--epochs N                  # Number of training epochs (default: 50)
--batch-size N             # Batch size (default: 32)
--lr FLOAT                 # Learning rate for new layers (default: 1e-3)
--clip-lr FLOAT            # Learning rate for CLIP (default: 1e-5)
--freeze-epochs N          # Epochs to keep CLIP frozen (default: 3)
--device {cuda|cpu}        # Device to use
--experiment-name NAME     # Experiment identifier
```

---

## 📈 Evaluation

### Evaluate on Test Set

```bash
python src/evaluate.py \
    --checkpoint checkpoints/mvsa_multiple_real_20251118_172701/best_model.pth \
    --clip-model openai/clip-vit-base-patch32 \
    --data-dir data/processed \
    --split test \
    --device cuda
```

---

## 📂 Project Structure

```
Multimodal-Sentiment-Analysis-Project-SML/
├── src/
│   ├── models/
│   │   ├── clip_text.py          # CLIP text encoder wrapper
│   │   ├── clip_image.py         # CLIP vision encoder wrapper
│   │   ├── cross_attention.py    # Bidirectional cross-attention
│   │   ├── fusion.py             # Modality gating
│   │   └── auxiliary.py          # Auxiliary decoder
│   ├── systems/
│   │   └── cdan_model.py         # Main CDAN model
│   ├── data/
│   │   ├── mvsa.py               # MVSA dataset loaders
│   │   └── datamodule.py         # PyTorch Lightning DataModule
│   ├── utils/
│   │   ├── metrics.py            # Evaluation metrics
│   │   └── logger.py             # Logging utilities
│   ├── train.py                  # Training script
│   ├── evaluate.py               # Evaluation script
│   └── predict.py                # Inference script
├── scripts/
│   ├── prepare_mvsa_multiple.py  # Dataset preparation
│   ├── download_clip_model.py    # Download CLIP offline
│   └── train_cdan.sh            # Training bash script
├── configs/
│   └── cdan.yaml                # Configuration file
├── data/
│   ├── processed/               # CSV files
│   └── raw/images/              # Image files
├── checkpoints/                 # Saved models
├── logs/                        # Training logs
└── README.md                    # This file
```

---

## 🎓 Citations

### CLIP
```bibtex
@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  booktitle={International Conference on Machine Learning},
  pages={8748--8763},
  year={2021},
  organization={PMLR}
}
```

### Cross-Modal Attention (ViLBERT)
```bibtex
@inproceedings{lu2019vilbert,
  title={Vilbert: Pretraining task-agnostic visiolinguistic representations},
  author={Lu, Jiasen and Batra, Dhruv and Parikh, Devi and Lee, Stefan},
  booktitle={Advances in Neural Information Processing Systems},
  pages={13--23},
  year={2019}
}
```

### MVSA Dataset
```bibtex
@inproceedings{niu2016mvsa,
  title={MVSA: A multilingual, multi-view sentiment analysis dataset},
  author={Niu, Teng and Zhu, Shiai and Pang, Lei and El Saddik, Abdulmotaleb},
  booktitle={Proceedings of the 2016 ACM on Multimedia Conference},
  pages={1034--1038},
  year={2016}
}
```

---

## 🙏 Acknowledgments

- **OpenAI** for the CLIP model
- **MVSA Dataset** creators for the multimodal sentiment analysis benchmark
- **PyTorch** and **Hugging Face** teams for excellent frameworks
- **NVIDIA** for GPU compute resources

---

**Last Updated**: November 18, 2025
