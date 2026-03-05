# Palme OCR — Low-Resource Document Digitization Pipeline

A fully offline, Dockerized OCR pipeline for digitizing scanned documents from the Olof Palme cold case archives. Built on [dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr) (1.7B parameters), a state-of-the-art vision-language model for multilingual document parsing.

## Quick Start

### 1. Build the Docker Image

> **Note:** The build step downloads model weights (~3.5 GB) and dependencies. Internet is required only during build.

```bash
docker build -t palme-ocr .
```

Build time: ~10–15 minutes depending on internet speed.

### 2. Run OCR on a Folder of PDFs

**With GPU (recommended):**
```bash
docker run --gpus all --rm \
  -v /absolute/path/to/pdfs:/data/input \
  -v /absolute/path/to/output:/data/output \
  palme-ocr \
  --input /data/input \
  --output /data/output
```

**Without GPU (CPU fallback):**
```bash
docker run --rm \
  -v /absolute/path/to/pdfs:/data/input \
  -v /absolute/path/to/output:/data/output \
  palme-ocr \
  --input /data/input \
  --output /data/output
```

**Example with the development set:**
```bash
mkdir -p results
docker run --gpus all --rm \
  -v $(pwd)/dev_set:/data/input \
  -v $(pwd)/results:/data/output \
  palme-ocr \
  --input /data/input \
  --output /data/output
```

### 3. Check Results

Each PDF produces corresponding output files:

```
results/
├── document_001.txt        # Plain text (reading order)
├── document_001.docx       # Formatted Word document
├── document_001_page1.raw.json  # Raw model output (for debugging)
└── ...
```

## Architecture

```
PDF Input
    │
    ▼
┌──────────────────────┐
│  PDF → Images         │  PyMuPDF, 150 DPI rendering
│  (pdf_processor.py)   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Adaptive Preprocess  │  Analyzes brightness, contrast, edge density,
│  (preprocess.py)      │  noise, skew → applies targeted fixes
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  dots.ocr VLM         │  1.7B param model, structured JSON output
│  (ocr_engine.py)      │  Layout detection + OCR in one pass
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Post-processing      │  Robust JSON parser → clean text + formatted docx
│  (postprocess.py)     │
└──────────┬───────────┘
           │
           ▼
   .txt + .docx Output
```

## Adaptive Preprocessing

Each page is analyzed before processing. The pipeline detects and applies only the relevant enhancements:

| Condition | Detection | Action |
|---|---|---|
| Dark scan | Mean brightness < 120 | CLAHE + gamma correction |
| Faded text | Contrast std < 40 | Aggressive CLAHE |
| Dense/small text | Edge density > 0.08 | Sharpening + fine CLAHE |
| Noisy scan | Laplacian variance > 1000 | Non-local means denoising |
| Skewed page | Hough line angle > 0.3° | Affine rotation correction |

## Device Auto-Detection

The pipeline automatically selects the optimal configuration:

| Environment | Attention | Dtype | Speed |
|---|---|---|---|
| NVIDIA GPU (CUDA) | flash_attention_2 (fallback: eager) | bfloat16 | ~15 sec/page |
| CPU | eager | bfloat16 | ~5 min/page |

No code changes needed — the same Docker image works on both.

## Running Without Docker (Local Development)

### Prerequisites
- Python 3.10+
- Git

### Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch
# For CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# For CPU only (Mac/Linux):
pip install torch torchvision

# Install dots.ocr
git clone https://github.com/rednote-hilab/dots.ocr.git dots_ocr_repo
cd dots_ocr_repo && pip install -e . && cd ..

# Install dependencies
pip install PyMuPDF opencv-python-headless Pillow python-docx huggingface-hub

# (Optional, CUDA only) Install flash attention for faster inference
pip install flash-attn --no-build-isolation

# Download model weights (~3.5 GB)
python -c "from huggingface_hub import snapshot_download; snapshot_download('rednote-hilab/dots.ocr', local_dir='./weights/DotsOCR')"
```

### Run

```bash
python src/main.py --input ./path/to/pdfs --output ./results --model-path ./weights/DotsOCR
```

## Project Structure

```
palme-ocr/
├── Dockerfile              # CUDA-enabled Docker build
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── src/
│   ├── main.py             # CLI entrypoint
│   ├── ocr_engine.py       # dots.ocr model (auto CUDA/CPU)
│   ├── pdf_processor.py    # PDF → image conversion
│   ├── preprocess.py       # Adaptive image enhancement
│   ├── postprocess.py      # JSON → clean text extraction
│   ├── docx_writer.py      # Structured JSON → Word document
│   └── reprocess.py        # Regenerate outputs from cached raw data
└── report.pdf              # Technical report
```

## Constraints Compliance

| Constraint | Status |
|---|---|
| Offline inference | ✅ All assets baked into Docker image |
| Model ≤ 3B params | ✅ dots.ocr = 1.7B parameters |
| Low-resource hardware | ✅ Runs on CPU or single consumer GPU |
| Python 3.9+ | ✅ Python 3.12 |
| Reproducible build | ✅ Dockerfile + pinned dependencies |

## Hardware Requirements

- **RAM:** 8 GB minimum, 16 GB recommended
- **Disk:** ~12 GB for Docker image (model weights baked in)
- **CPU:** Any modern x86_64 processor
- **GPU:** Optional — NVIDIA GPU with CUDA auto-detected for ~20x faster inference

## Team

- **Team Name:** Butterscotch
- **Members:** Madusha Thilakarathna, Kavindi Peiris