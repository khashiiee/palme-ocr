#!/bin/bash
# Local development setup for Mac M3 Pro (Apple Silicon)
# Run this ONCE to set up your environment, then use run_local.sh

set -e

echo "=== Palme OCR — Local Dev Setup (Apple Silicon) ==="

# Create conda environment
echo "[1/4] Creating conda environment..."
conda create -n palme-ocr python=3.12 -y
eval "$(conda shell.bash hook)"
conda activate palme-ocr

# Install PyTorch for Apple Silicon (MPS support)
echo "[2/4] Installing PyTorch (MPS)..."
pip install torch torchvision

# Clone and install dots.ocr
echo "[3/4] Installing dots.ocr..."
if [ ! -d "dots_ocr_repo" ]; then
    git clone https://github.com/rednote-hilab/dots.ocr.git dots_ocr_repo
fi
cd dots_ocr_repo && pip install -e . && cd ..

# Install other dependencies
pip install PyMuPDF opencv-python-headless Pillow huggingface-hub

# Download model weights
echo "[4/4] Downloading model weights (~3.5 GB)..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('rednote-hilab/dots.ocr', local_dir='./weights/DotsOCR')
"

echo ""
echo "=== Setup complete! ==="
echo "Activate with: conda activate palme-ocr"
echo "Run with:      python src/main.py --input ./dev_set --output ./results --model-path ./weights/DotsOCR"
