# Base image
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1. Install System Packages
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget unzip curl vim tmux nano \
    ffmpeg libgl1 libglib2.0-0 python3-opencv \
    libgl1-mesa-glx libglib2.0-0 x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2. Install Required Packages
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
RUN pip install --no-cache-dir "numpy==1.26.4" "matplotlib<3.8"

RUN pip install \
    scipy pandas pyyaml \
    torchmetrics lightning==2.4.0 accelerate==1.0.1 \
    timm==1.0.9 einops==0.8.0 \
    transformers==4.44.2 sentencepiece==0.2.0 \
    opencv-python-headless==4.10.0.84 \
    albumentations==1.4.14 pycocotools==2.0.8

# nuScenes SDK
RUN pip install nuscenes-devkit==1.1.11 mlflow==2.16.0

RUN pip install --no-build-isolation flash-attn==2.6.3 || true

# numpy and matplotlib are installed again to avoid version conflicts
RUN pip install --no-cache-dir "numpy==1.26.4" "matplotlib<3.8"

ENV TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    CUDA_DEVICE_MAX_CONNECTIONS=1

WORKDIR /workspace