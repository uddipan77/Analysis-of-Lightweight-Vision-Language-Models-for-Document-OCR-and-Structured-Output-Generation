# =============================================================================
# Dockerfile — Gemma-3 4B Multi-Dataset OCR UI
# =============================================================================
# Bakes in the base model (4-bit) and LoRA adapter so the image is self-contained.
# Requires NVIDIA GPU + nvidia-container-toolkit on the host to run.
#
# BUILD (from workspace root):
#   docker build -t uddipanbb/gemma-ocr-ui:latest -f Dockerfile .
#
# RUN:
#   docker run --rm --gpus all -p 8000:8000 uddipanbb/gemma-ocr-ui:latest
#
# Then open http://localhost:8000/ui
# =============================================================================

FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ──
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
        git \
        wget \
        curl \
        gcc \
        g++ \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ── Working directory ──
WORKDIR /app

# ── Install Python dependencies (cached layer) ──
COPY requirements.docker.txt /app/requirements.docker.txt
RUN pip install --no-cache-dir -r /app/requirements.docker.txt

# ── Copy base model ──
# Source: gemma_model/models--unsloth--gemma-3-4b-it-unsloth-bnb-4bit/snapshots/<hash>/
COPY gemma_model/models--unsloth--gemma-3-4b-it-unsloth-bnb-4bit/snapshots/316726ca0bd24aa323bfaf86e8a379ee1176d1fe/ \
     /app/models/base_model/

# ── Copy LoRA adapter ──
# Source: UI_main/gemma checkpoints used for ui/.../best_model/
COPY ["UI_main/gemma checkpoints used for ui/run_20251216_010403_gemma3_multidataset/best_model/", \
      "/app/models/lora_adapter/"]

# ── Copy application code ──
COPY UI_main/gemma_ui_docker.py /app/gemma_ui_docker.py

# ── Environment variables ──
ENV BASE_MODEL_PATH=/app/models/base_model
ENV FINETUNED_ADAPTER_PATH=/app/models/lora_adapter
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV PORT=8000

# ── Expose port ──
EXPOSE 8000

# ── Health check ──
HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Run ──
CMD ["python", "gemma_ui_docker.py"]
