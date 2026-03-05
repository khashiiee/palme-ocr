FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir \
    torch==2.7.0 \
    torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Clone and install dots.ocr
RUN git clone https://github.com/rednote-hilab/dots.ocr.git /app/dots_ocr_repo && \
    cd /app/dots_ocr_repo && \
    pip install --no-cache-dir -e .

# Install additional dependencies
RUN pip install --no-cache-dir \
    PyMuPDF>=1.24.0 \
    opencv-python-headless>=4.9.0 \
    Pillow>=10.0.0 \
    python-docx>=1.1.0 \
    huggingface-hub>=0.20.0

# Download model weights at build time (no periods in path!)
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('rednote-hilab/dots.ocr', local_dir='/app/weights/DotsOCR')"

# Copy source code
COPY src/ /app/src/

# Set model path on PYTHONPATH for trust_remote_code
ENV PYTHONPATH="/app/weights:${PYTHONPATH}"

ENTRYPOINT ["python", "/app/src/main.py"]