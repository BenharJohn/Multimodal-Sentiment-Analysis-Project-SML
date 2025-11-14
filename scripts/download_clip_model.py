#!/usr/bin/env python3
"""
Download CLIP model from HuggingFace and cache it locally.
This script downloads the model once so training doesn't need internet.
"""

import os
from pathlib import Path

print("="*60)
print("Downloading CLIP Model for CDAN Training")
print("="*60)
print()

# Model to download
model_name = "openai/clip-vit-base-patch32"
print(f"Model: {model_name}")
print(f"Expected size: ~605 MB (PyTorch version)")
print()

try:
    from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
    from transformers import AutoImageProcessor

    print("Step 1/4: Downloading CLIP model...")
    model = CLIPModel.from_pretrained(model_name)
    print("✅ Model downloaded")

    print("\nStep 2/4: Downloading CLIP processor...")
    processor = CLIPProcessor.from_pretrained(model_name)
    print("✅ Processor downloaded")

    print("\nStep 3/4: Downloading tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    print("✅ Tokenizer downloaded")

    print("\nStep 4/4: Verifying downloads...")
    # Test that everything works
    import torch
    test_text = ["a photo of a cat"]
    test_inputs = processor(text=test_text, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**test_inputs)
    print("✅ Verification successful")

    print("\n" + "="*60)
    print("SUCCESS! CLIP model cached and ready to use")
    print("="*60)
    print()

    # Show cache location
    from transformers.utils import TRANSFORMERS_CACHE
    cache_dir = os.environ.get('TRANSFORMERS_CACHE',
                                os.environ.get('HF_HOME',
                                              Path.home() / '.cache' / 'huggingface'))
    print(f"Cache location: {cache_dir}")
    print()
    print("You can now run training without internet!")
    print("Command: python src/train.py --epochs 2 --batch-size 4 --device cpu")

except ImportError as e:
    print(f"❌ Error: {e}")
    print("Please install: pip install transformers torch")
    exit(1)
except Exception as e:
    print(f"❌ Download failed: {e}")
    print()
    print("Possible issues:")
    print("1. No internet connection")
    print("2. HuggingFace servers temporarily down")
    print("3. Firewall blocking connection")
    print()
    print("Try again or download manually from:")
    print("https://huggingface.co/openai/clip-vit-base-patch32")
    exit(1)
