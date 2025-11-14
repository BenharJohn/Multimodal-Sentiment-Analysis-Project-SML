#!/bin/bash
# Manual CLIP Model Download Script
# Run this on a machine with internet, then transfer files to offline environment

echo "============================================================"
echo "Manual CLIP Model Download"
echo "============================================================"
echo ""
echo "This script downloads CLIP model files from HuggingFace."
echo "You need: wget or curl, and ~700 MB free space"
echo ""

# Create download directory
MODEL_DIR="clip-vit-base-patch32"
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

echo "Downloading model files..."
echo ""

# Base URL
BASE_URL="https://huggingface.co/openai/clip-vit-base-patch32/resolve/main"

# Required files for PyTorch version
FILES=(
    "config.json"
    "preprocessor_config.json"
    "vocab.json"
    "merges.txt"
    "tokenizer.json"
    "tokenizer_config.json"
    "special_tokens_map.json"
    "pytorch_model.bin"  # ~605 MB - the main model weights
)

# Download each file
for file in "${FILES[@]}"; do
    echo "Downloading $file..."

    # Try wget first, fallback to curl
    if command -v wget &> /dev/null; then
        wget -q --show-progress "$BASE_URL/$file" -O "$file"
    elif command -v curl &> /dev/null; then
        curl -L -o "$file" "$BASE_URL/$file"
    else
        echo "Error: Neither wget nor curl found. Please install one."
        exit 1
    fi

    if [ $? -eq 0 ]; then
        echo "✅ $file downloaded"
    else
        echo "❌ Failed to download $file"
        exit 1
    fi
done

echo ""
echo "============================================================"
echo "SUCCESS! All files downloaded to: $MODEL_DIR/"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Transfer this folder to your offline environment"
echo "2. Place in HuggingFace cache directory:"
echo "   ~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/"
echo ""
echo "Or use in code:"
echo "   model = CLIPModel.from_pretrained('./clip-vit-base-patch32')"
echo ""
