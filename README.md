# Palme OCR — Low-Resource Document Digitization Pipeline

A fully offline, Dockerized OCR pipeline for digitizing scanned documents from the Olof Palme cold case archives. Built on [dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr) (1.7B parameters), a state-of-the-art vision-language model for document parsing.

## Quick Start

### 1. Build the Docker Image

> **Note:** The build step downloads model weights (~3.5 GB) and dependencies. Internet is required only during build.

```bash
docker build -t palme-ocr .
```

Build time: ~10–15 minutes depending on internet speed.

### 2. Run OCR on a Folder of PDFs

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
docker run --rm \
  -v $(pwd)/dev_set:/data/input \
  -v $(pwd)/results:/data/output \
  palme-ocr \
  --input /data/input \
  --output /data/output
```

### 3. Check Results

Each PDF produces a corresponding `.txt` file in the output directory:

```
results/
├── document_001.txt
├── document_002.txt
└── ...
```

## Architecture Overview

```
PDF Input
    │
    ▼
┌──────────────────┐
│  PDF → Images     │  PyMuPDF, 300 DPI rendering
│  (pdf_processor)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Pre-processing   │  Deskew, CLAHE contrast enhancement,
│  (preprocess)     │  adaptive thresholding (OpenCV)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  dots.ocr VLM     │  1.7B param vision-language model
│  (ocr_engine)     │  Layout detection + OCR in one pass
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Post-processing  │  JSON parsing → clean plain text
│  (postprocess)    │  in correct reading order
└────────┬─────────┘
         │
         ▼
   .txt Output
```

## Technical Details

- **Model:** dots.ocr (1.7B LLM parameters, MIT license)
- **Inference:** CPU by default; NVIDIA GPU auto-detected if available
- **Pre-processing:** OpenCV-based deskew, CLAHE contrast enhancement, noise reduction
- **Output format:** Plain text files preserving document reading order

## Hardware Requirements

- **RAM:** 8 GB minimum, 16 GB recommended
- **Disk:** ~8 GB for the Docker image (model weights baked in)
- **CPU:** Any modern x86_64 processor
- **GPU:** Optional — NVIDIA GPU with CUDA will be auto-detected for faster inference

## Project Structure

```
palme-ocr/
├── Dockerfile
├── README.md
├── requirements.txt
├── src/
│   ├── main.py           # CLI entrypoint
│   ├── ocr_engine.py     # Model loading & inference
│   ├── pdf_processor.py  # PDF → image conversion
│   ├── preprocess.py     # Image enhancement pipeline
│   └── postprocess.py    # Structured JSON → clean text
└── report.pdf            # Technical report
```

## Constraints Compliance

| Constraint | Status |
|---|---|
| Offline inference | ✅ All assets baked into Docker image |
| Model ≤ 3B params | ✅ dots.ocr = 1.7B parameters |
| Low-resource hardware | ✅ Runs on CPU, ~6 GB RAM during inference |
| Python 3.9+ | ✅ Python 3.11 |
| Reproducible build | ✅ Pinned requirements.txt + Dockerfile |

## Team

- **Team Name:** TODO
- **Members:** TODO
