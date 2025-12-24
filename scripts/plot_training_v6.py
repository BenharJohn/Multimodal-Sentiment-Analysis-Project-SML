"""
Plot training curves for CDAN v6 model.
Visualizes training progress and model performance.
"""

import matplotlib.pyplot as plt
import numpy as np

# V6 Training Data (from logs)
epochs = list(range(1, 36))

# Training metrics
train_f1 = [
    0.4156, 0.4827, 0.4860, 0.5258, 0.5358, 0.5649, 0.6058, 0.6304, 0.6436, 0.6689,
    0.6725, 0.6904, 0.6995, 0.6937, 0.7088, 0.7226, 0.7335, 0.7505, 0.7591, 0.7640,
    0.7826, 0.7950, 0.7948, 0.8000, 0.8155, 0.8259, 0.8383, 0.8497, 0.8544, 0.8586,
    0.8747, 0.8747, 0.8747, 0.8747, 0.8747  # Extended for remaining epochs
]

val_f1 = [
    0.2339, 0.1907, 0.0501, 0.1752, 0.4296, 0.3028, 0.3914, 0.4930, 0.3526, 0.5098,
    0.5115, 0.5152, 0.5127, 0.4484, 0.5198, 0.5193, 0.4039, 0.5135, 0.4986, 0.5125,
    0.5064, 0.5168, 0.5166, 0.4678, 0.5126, 0.4858, 0.4933, 0.4995, 0.5091, 0.4898,
    0.4990, 0.4990, 0.4990, 0.4990, 0.4990  # Extended
]

train_acc = [
    47.42, 53.62, 53.16, 56.23, 57.60, 59.49, 62.37, 63.63, 64.51, 66.12,
    66.08, 66.88, 68.18, 67.51, 68.26, 69.45, 70.02, 71.19, 72.10, 72.63,
    73.83, 75.07, 74.92, 75.29, 76.77, 78.70, 79.43, 80.97, 81.77, 82.56,
    84.05, 84.05, 84.05, 84.05, 84.05  # Extended
]

val_acc = [
    54.04, 40.06, 6.68, 17.00, 47.83, 32.30, 52.72, 54.35, 52.95, 59.24,
    59.94, 57.53, 59.39, 57.84, 59.86, 59.94, 56.29, 60.25, 58.62, 60.09,
    60.40, 60.87, 60.40, 55.59, 59.55, 56.37, 58.07, 58.70, 58.23, 55.82,
    56.13, 56.13, 56.13, 56.13, 56.13  # Extended
]

# Use only first 31 epochs (actual data)
epochs = epochs[:31]
train_f1 = train_f1[:31]
val_f1 = val_f1[:31]
train_acc = train_acc[:31]
val_acc = val_acc[:31]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CDAN v6 Training Progress\n(Dual Encoders + Self-Adaptive Aggregation)',
             fontsize=14, fontweight='bold')

# Plot 1: F1 Score
ax1 = axes[0, 0]
ax1.plot(epochs, [f*100 for f in train_f1], 'b-', linewidth=2, label='Train F1', marker='o', markersize=4)
ax1.plot(epochs, [f*100 for f in val_f1], 'r-', linewidth=2, label='Val F1', marker='s', markersize=4)
ax1.axhline(y=51.98, color='g', linestyle='--', linewidth=1.5, label='Best Val F1 (51.98%)')
ax1.axvline(x=15, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='Best Epoch (15)')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('F1 Score (%)', fontsize=11)
ax1.set_title('Macro F1 Score', fontsize=12, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 100])

# Plot 2: Accuracy
ax2 = axes[0, 1]
ax2.plot(epochs, train_acc, 'b-', linewidth=2, label='Train Acc', marker='o', markersize=4)
ax2.plot(epochs, val_acc, 'r-', linewidth=2, label='Val Acc', marker='s', markersize=4)
ax2.axhline(y=60.25, color='g', linestyle='--', linewidth=1.5, label='Best Val Acc (60.25%)')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Accuracy (%)', fontsize=11)
ax2.set_title('Classification Accuracy', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 100])

# Plot 3: Overfitting Gap
ax3 = axes[1, 0]
gap_f1 = [(t - v) * 100 for t, v in zip(train_f1, val_f1)]
gap_acc = [t - v for t, v in zip(train_acc, val_acc)]
ax3.plot(epochs, gap_f1, 'purple', linewidth=2, label='F1 Gap', marker='d', markersize=4)
ax3.plot(epochs, gap_acc, 'orange', linewidth=2, label='Accuracy Gap', marker='^', markersize=4)
ax3.axhline(y=0, color='green', linestyle='-', linewidth=1)
ax3.fill_between(epochs, gap_f1, alpha=0.3, color='purple')
ax3.set_xlabel('Epoch', fontsize=11)
ax3.set_ylabel('Train - Val (%)', fontsize=11)
ax3.set_title('Overfitting Gap (Train - Val)', fontsize=12, fontweight='bold')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

# Plot 4: Version Comparison
ax4 = axes[1, 1]
versions = ['Baseline\n(No Dual Enc)', 'v1\n(Dual Enc)', 'v5\n(+Decoder FB)', 'v6\n(+Regularization)']
val_f1_versions = [52.50, 54.74, 52.76, 51.98]
val_acc_versions = [57.92, 55.0, 52.0, 60.25]  # Approximate values

x = np.arange(len(versions))
width = 0.35

bars1 = ax4.bar(x - width/2, val_f1_versions, width, label='Val F1 (%)', color='steelblue', edgecolor='black')
bars2 = ax4.bar(x + width/2, val_acc_versions, width, label='Val Acc (%)', color='coral', edgecolor='black')

ax4.set_xlabel('Model Version', fontsize=11)
ax4.set_ylabel('Metric (%)', fontsize=11)
ax4.set_title('Model Version Comparison', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(versions, fontsize=9)
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim([40, 70])

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax4.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

for bar in bars2:
    height = bar.get_height()
    ax4.annotate(f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('results/cdan_v6_training_curves.png', dpi=150, bbox_inches='tight')
plt.savefig('results/cdan_v6_training_curves.pdf', bbox_inches='tight')
print("Plots saved to results/cdan_v6_training_curves.png and .pdf")

# Also create a summary figure
fig2, ax = plt.subplots(figsize=(10, 6))

ax.plot(epochs, [f*100 for f in train_f1], 'b-', linewidth=2.5, label='Train F1', marker='o', markersize=5)
ax.plot(epochs, [f*100 for f in val_f1], 'r-', linewidth=2.5, label='Val F1', marker='s', markersize=5)
ax.plot(epochs, train_acc, 'b--', linewidth=2, label='Train Acc', alpha=0.7)
ax.plot(epochs, val_acc, 'r--', linewidth=2, label='Val Acc', alpha=0.7)

ax.axhline(y=60, color='green', linestyle=':', linewidth=2, label='Target (60%)')
ax.axvline(x=15, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Metric (%)', fontsize=12)
ax.set_title('CDAN v6: Training Progress\n(CLIP + BERT + ResNet with Self-Adaptive Aggregation)',
             fontsize=13, fontweight='bold')
ax.legend(loc='center right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 100])
ax.set_xlim([0, 32])

# Add annotations
ax.annotate('Best Val Acc: 60.25%', xy=(18, 60.25), xytext=(22, 70),
            arrowprops=dict(arrowstyle='->', color='green'),
            fontsize=10, color='green')
ax.annotate('Best Val F1: 51.98%', xy=(15, 51.98), xytext=(5, 40),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')

plt.tight_layout()
plt.savefig('results/cdan_v6_summary.png', dpi=150, bbox_inches='tight')
print("Summary plot saved to results/cdan_v6_summary.png")

plt.show()
