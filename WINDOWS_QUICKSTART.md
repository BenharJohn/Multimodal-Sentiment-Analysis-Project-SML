# 🪟 Windows Quick Start Guide

Complete setup guide for running CDAN on Windows 10/11.

---

## ✅ **Prerequisites**

- Windows 10/11
- Python 3.8+ (Download from [python.org](https://www.python.org/downloads/))
- Git for Windows (Download from [git-scm.com](https://git-scm.com/download/win))
- ~2 GB free disk space (for model + dependencies)

---

## 🚀 **Complete Setup (5 Steps)**

### **Step 1: Clone Repository**

```powershell
# Open PowerShell and navigate to your desired directory
cd F:\  # Or any drive you prefer

# Clone the repository
git clone https://github.com/BenharJohn/Multimodal-Sentiment-Analysis-Project-SML.git
cd Multimodal-Sentiment-Analysis-Project-SML
```

---

### **Step 2: Install PyTorch (Choose ONE option)**

#### **Option A: Using Conda (Recommended for Windows)**

```powershell
# Install Miniconda first if you don't have it
# Download from: https://docs.conda.io/en/latest/miniconda.html

# Create environment
conda create -n cdan python=3.10 -y
conda activate cdan

# Install PyTorch (CPU version)
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install other dependencies
pip install -r requirements.txt
```

**✅ This is the most reliable option for Windows!**

#### **Option B: Using venv + pip**

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install PyTorch CPU version (with proper Windows DLLs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

**⚠️ If you see DLL errors later, switch to Option A (conda).**

---

### **Step 3: Download CLIP Model**

#### **Method 1: Automatic Download (Easiest)**

```powershell
# Run the Python download script
python scripts/download_clip_model.py
```

This downloads ~610 MB and caches to:
```
C:\Users\<YourUsername>\.cache\huggingface\hub\models--openai--clip-vit-base-patch32\
```

#### **Method 2: Manual Download (If automatic fails)**

```powershell
# Download using PowerShell
$base_url = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main"
$files = @(
    "config.json",
    "preprocessor_config.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "pytorch_model.bin"
)

New-Item -ItemType Directory -Force -Path "clip-vit-base-patch32"
Set-Location "clip-vit-base-patch32"

foreach ($file in $files) {
    Write-Host "Downloading $file..."
    Invoke-WebRequest -Uri "$base_url/$file" -OutFile $file
}

# Move to cache location
$cache_dir = "$env:USERPROFILE\.cache\huggingface\hub\models--openai--clip-vit-base-patch32\snapshots\main"
New-Item -ItemType Directory -Force -Path $cache_dir
Copy-Item * $cache_dir -Force
```

**Verify download:**
```powershell
python -c "from transformers import CLIPModel; m = CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); print('✅ Model loaded!')"
```

---

### **Step 4: Prepare Data**

#### **For Testing - Create Dummy Dataset:**

```powershell
# PowerShell version of prepare_data.sh
python -c @"
import os, csv, random
from pathlib import Path
from PIL import Image
import numpy as np

# Create directories
for d in ['data/raw/images', 'data/processed']:
    Path(d).mkdir(parents=True, exist_ok=True)

# Generate 100 dummy images
print('Generating 100 dummy images...')
for i in range(100):
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img.save(f'data/raw/images/img_{i:03d}.jpg')

# Create dataset CSV
labels = ['positive', 'negative', 'neutral']
samples = []
for i in range(100):
    samples.append({
        'id': i,
        'image_path': f'data/raw/images/img_{i:03d}.jpg',
        'text': f'This is sample text {i}',
        'label': random.choice(labels)
    })

# Shuffle and split
random.shuffle(samples)
train = samples[:70]
val = samples[70:85]
test = samples[85:]

# Write CSVs
for name, data in [('train', train), ('val', val), ('test', test)]:
    with open(f'data/processed/{name}.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'image_path', 'text', 'label'])
        writer.writeheader()
        writer.writerows(data)

print(f'✅ Created: {len(train)} train, {len(val)} val, {len(test)} test')
"@
```

#### **For Real Training - Download MVSA Datasets:**

See `README.md` section "Dataset Preparation" for download links and instructions.

---

### **Step 5: Run Training**

#### **Quick Test (2 epochs on dummy data):**

```powershell
python src/train.py --epochs 2 --batch-size 4 --device cpu --num-workers 0
```

**Expected output:**
```
============================================================
Starting CDAN Training
============================================================

Loading CLIP model...
✅ CLIP model loaded

Loading datasets...
✅ Train: 70 samples
✅ Val: 15 samples

Epoch 1/2:
  Training: 100%|████████| Loss: 1.0923, Acc: 42.86%
  Validation: Loss: 1.0654, Acc: 46.67%, F1: 0.4123

Epoch 2/2:
  Training: 100%|████████| Loss: 1.0234, Acc: 51.43%
  Validation: Loss: 0.9987, Acc: 53.33%, F1: 0.4789

✅ Training complete!
Saved checkpoint: checkpoints/cdan_mvsa_single_YYYYMMDD_HHMMSS/best_model.pth
```

#### **Full Training (30 epochs, real data):**

```powershell
python src/train.py `
    --config configs/cdan.yaml `
    --epochs 30 `
    --batch-size 32 `
    --device cuda `
    --num-workers 4
```

---

## 🔧 **Troubleshooting**

### **Issue 1: DLL Initialization Failed (c10.dll, torch.dll)**

**Error:**
```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed
```

**Solutions:**

1. **Use Conda (Best fix):**
   ```powershell
   # Deactivate venv if active
   deactivate

   # Install conda and use it
   conda create -n cdan python=3.10
   conda activate cdan
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   pip install -r requirements.txt
   ```

2. **Reinstall PyTorch in venv:**
   ```powershell
   pip uninstall torch torchvision torchaudio -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Install Visual C++ Redistributables:**
   - Download: [VC++ 2015-2022 Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe)
   - Install and restart PowerShell

---

### **Issue 2: CLIP Model Not Found**

**Error:**
```
OSError: We couldn't connect to 'https://huggingface.co' to load the files
```

**Solution:**
Model not cached. Run:
```powershell
python scripts/download_clip_model.py
```

Check cache location:
```powershell
python -c "from pathlib import Path; print(Path.home() / '.cache' / 'huggingface')"
```

---

### **Issue 3: Permission Denied on Scripts**

Windows doesn't need `chmod +x`. Just run scripts with `python`:
```powershell
# Don't do: bash scripts/train_cdan.sh
# Instead:
python src/train.py --epochs 2 --batch-size 4
```

---

### **Issue 4: Import Errors**

```powershell
# Verify installation
pip list | Select-String "torch|transformers|pillow|numpy"

# Should see:
# torch        2.x.x
# torchvision  0.x.x
# transformers 4.x.x
# numpy        1.x.x
# pillow       10.x.x
```

If missing, reinstall:
```powershell
pip install -r requirements.txt --force-reinstall
```

---

## 📊 **Evaluation**

After training, evaluate the model:

```powershell
python src/evaluate.py `
    --checkpoint checkpoints/cdan_mvsa_single_*/best_model.pth `
    --save-predictions `
    --visualize-attention
```

Output:
- `results/metrics.json` - Accuracy, F1, precision, recall
- `results/predictions.csv` - Per-sample predictions
- `results/confusion_matrix.png` - Confusion matrix heatmap
- `results/attention_*.png` - Attention visualizations

---

## 🎯 **Expected Performance**

| Dataset | Accuracy | Macro-F1 | Training Time |
|---------|----------|----------|---------------|
| MVSA-Single | ≥78% | ≥73% | ~2-3h (GPU) |
| MVSA-Multiple | ≥75% | ≥70% | ~2-3h (GPU) |
| Dummy Data | ~40-60% | ~35-55% | ~5 min (CPU) |

**Note:** Dummy data is random, so metrics will be low. Use real MVSA data for actual performance.

---

## 📁 **File Locations (Windows)**

| Item | Location |
|------|----------|
| **Project** | `F:\SML\Multimodal-Sentiment-Analysis-Project-SML\` |
| **CLIP Cache** | `C:\Users\<Username>\.cache\huggingface\hub\` |
| **Checkpoints** | `checkpoints\cdan_mvsa_single_*\` |
| **Results** | `results\` |
| **Logs** | `logs\` |

---

## 💡 **Tips for Windows Users**

1. **Use PowerShell** (not CMD) - Better for Python development
2. **Use Conda** - More reliable than venv on Windows
3. **CPU vs GPU:**
   - CPU: Use `--device cpu` (slower but works everywhere)
   - GPU: Requires CUDA-enabled NVIDIA GPU + CUDA toolkit
4. **Backslashes:** Windows uses `\` but Python accepts `/` in paths
5. **Long paths:** Enable long path support if you see path errors:
   ```powershell
   # Run PowerShell as Administrator
   New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
   ```

---

## ✅ **Verification Checklist**

After setup, verify everything works:

- [ ] Python 3.8+ installed: `python --version`
- [ ] PyTorch works: `python -c "import torch; print(torch.__version__)"`
- [ ] CLIP cached: `python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"`
- [ ] Dependencies installed: `pip list | Select-String "transformers"`
- [ ] Dummy data created: `Test-Path data\processed\train.csv`
- [ ] Training runs: `python src/train.py --epochs 1 --batch-size 4 --device cpu`

---

## 🆘 **Still Having Issues?**

1. Check `README.md` for detailed documentation
2. See `OFFLINE_SETUP.md` for offline model setup
3. Check `CHECKLIST_REPORT.md` for validation details
4. Open an issue on GitHub with:
   - Windows version
   - Python version (`python --version`)
   - PyTorch version (`python -c "import torch; print(torch.__version__)"`)
   - Full error message

---

## 📚 **Next Steps**

1. **Run quick test** on dummy data (5 min)
2. **Download real MVSA datasets** (see README.md)
3. **Train full model** (30 epochs, 2-3h)
4. **Evaluate and visualize** results
5. **Compare with baselines** (reported in paper)

---

**Happy Training! 🎉**

*For Linux/Mac users, see `README.md` for bash-based instructions.*
