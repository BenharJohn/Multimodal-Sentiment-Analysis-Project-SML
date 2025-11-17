# 🔌 Offline Setup Guide for CDAN Training

This guide helps you run the CDAN model in environments **without internet access**.

---

## 📊 **Understanding the Model Size**

From HuggingFace: `openai/clip-vit-base-patch32` total: **1.82 GB**

**Breakdown:**
- `pytorch_model.bin`: **605 MB** ← We only need this!
- `tf_model.h5`: 606 MB (TensorFlow - not needed)
- `flax_model.msgpack`: 605 MB (Flax/JAX - not needed)
- Config files: ~5 MB

**For PyTorch training, you only need ~610 MB total.**

---

## 🎯 **Option 1: Download in Code (RECOMMENDED)**

If you have internet in your training environment:

```bash
# The model auto-downloads on first use
python src/train.py --epochs 30 --batch-size 32

# Downloads to: ~/.cache/huggingface/hub/
# Only happens ONCE, then cached forever
```

**That's it!** No manual steps needed.

---

## 🎯 **Option 2: Pre-Download for Offline Use**

If you need to prepare offline:

### **A. Using Python (Easiest)**

```bash
# On a machine with internet:
python scripts/download_clip_model.py

# This caches to: ~/.cache/huggingface/
# Then copy the cache folder to your offline machine
```

### **B. Manual Download (Full Control)**

**Step 1:** Run on machine with internet:

```bash
bash scripts/download_clip_manual.sh
```

This downloads 8 files (~610 MB total) to `clip-vit-base-patch32/` folder.

**Step 2:** Transfer files to offline machine and place in:

```
~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/snapshots/XXX/
```

Where `XXX` is any snapshot ID (or use in code directly).

---

## 📁 **Cache Directory Structure**

HuggingFace caches models here:

**Linux/Mac:**
```
~/.cache/huggingface/hub/
└── models--openai--clip-vit-base-patch32/
    ├── refs/
    │   └── main
    └── snapshots/
        └── <commit-hash>/
            ├── config.json
            ├── pytorch_model.bin        # ← 605 MB
            ├── preprocessor_config.json
            ├── tokenizer.json
            ├── tokenizer_config.json
            ├── vocab.json
            ├── merges.txt
            └── special_tokens_map.json
```

**Windows:**
```
C:\Users\<username>\.cache\huggingface\hub\
└── models--openai--clip-vit-base-patch32\
    ├── refs\
    │   └── main
    └── snapshots\
        └── <commit-hash>\
            ├── config.json
            ├── pytorch_model.bin        # ← 605 MB
            ├── (... same files as above)
```

**To check your cache location:**
```python
import os
from pathlib import Path
cache = os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface')
print(f"Cache: {cache}")
```

---

## 🔧 **Alternative: Use Local Model Path**

Download files, then modify `src/models/clip_text.py` and `clip_image.py`:

```python
# Instead of:
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Use:
model = CLIPModel.from_pretrained("./path/to/clip-vit-base-patch32")
```

---

## 📥 **Direct Download Links** (if scripts fail)

Download individual files from HuggingFace:

**Required Files:**
1. **config.json** (4 KB)
   - https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json

2. **pytorch_model.bin** (605 MB) ← MAIN MODEL
   - https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin

3. **preprocessor_config.json** (316 B)
   - https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/preprocessor_config.json

4. **tokenizer.json** (2.2 MB)
   - https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer.json

5. **tokenizer_config.json** (592 B)
   - https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer_config.json

6. **vocab.json** (862 KB)
   - https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/vocab.json

7. **merges.txt** (525 KB)
   - https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/merges.txt

8. **special_tokens_map.json** (389 B)
   - https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/special_tokens_map.json

**Total: ~610 MB**

---

## ⚡ **Quick Commands**

### **Linux/Mac - Download with wget:**
```bash
wget https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin
wget https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json
# ... (repeat for all 8 files)
```

### **Linux/Mac - Download with curl:**
```bash
curl -L -O https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin
curl -L -O https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json
# ... (repeat for all 8 files)
```

### **Linux/Mac - Download all at once:**
```bash
bash scripts/download_clip_manual.sh
```

### **Windows PowerShell - Download all files:**
```powershell
# Download using PowerShell
$base_url = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main"
$files = @("config.json", "preprocessor_config.json", "vocab.json", "merges.txt",
           "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
           "pytorch_model.bin")

New-Item -ItemType Directory -Force -Path "clip-vit-base-patch32"
Set-Location "clip-vit-base-patch32"

foreach ($file in $files) {
    Write-Host "Downloading $file..."
    Invoke-WebRequest -Uri "$base_url/$file" -OutFile $file
}
```

### **Windows - Using Python script (Easiest):**
```powershell
python scripts/download_clip_model.py
```

---

## ✅ **Verify Download**

After downloading, test:

```python
from transformers import CLIPModel

# Test loading
model = CLIPModel.from_pretrained("./clip-vit-base-patch32")
print("✅ Model loaded successfully!")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

Should output: `Parameters: 151,277,313` (151M parameters)

---

## 🚀 **Then Train Normally**

Once model is cached/downloaded:

```bash
# Quick test (2 epochs, dummy data)
python src/train.py --epochs 2 --batch-size 4 --device cpu

# Full training (real MVSA data)
bash scripts/train_cdan.sh --epochs 30
```

---

## 🔍 **Troubleshooting**

### **Issue: "Can't load model"**
**Solution:**
- Check cache directory exists
- Verify all 8 files are present
- Check file permissions

### **Issue: "OSError: Can't load image processor"**
**Solution:**
- Ensure `preprocessor_config.json` is in the same folder
- Try: `export TRANSFORMERS_CACHE=/path/to/cache`

### **Issue: Model too large**
**Solution:**
- You only need PyTorch version (~610 MB)
- Skip TensorFlow/Flax files (saves 1.2 GB)

### **Issue: Windows DLL Error - "c10.dll initialization failed"**
**Solution (Windows users):**
This occurs when PyTorch is missing Visual C++ dependencies.

**Option 1: Use conda (Recommended)**
```powershell
# Deactivate any venv first
deactivate

# Install PyTorch via conda
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

**Option 2: Reinstall PyTorch in venv**
```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Option 3: Install Visual C++ Redistributables**
- Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Install and restart terminal

---

## 📝 **Summary**

| Method | Pros | Cons |
|--------|------|------|
| **Auto-download** | Easiest, one command | Needs internet during training |
| **Pre-download script** | Automated, cacheable | Requires internet once |
| **Manual download** | Full control, verifiable | More steps |

**Recommendation:** Use auto-download if you have internet. Otherwise, use the manual script once and cache.

---

*See also:*
- `scripts/download_clip_model.py` - Automated download
- `scripts/download_clip_manual.sh` - Manual download script
- HuggingFace docs: https://huggingface.co/docs/transformers/installation#offline-mode
