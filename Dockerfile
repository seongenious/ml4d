# Base image: NVIDIA CUDA 11.8 + cuDNN 8 + Ubuntu 20.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1. Install System Packages
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 python3.8-venv python3.8-dev python3-pip \
    build-essential wget curl git unzip vim tmux nano ffmpeg \
    libgl1-mesa-glx libglib2.0-0 x11-apps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2. Install Required Packages
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Install numpy
RUN pip install numpy==1.23.5

# nuScenes SDK
RUN pip install nuscenes-devkit

# Waymo Open Dataset
RUN pip install waymo-open-dataset-tf-2-12-0==1.6.7

# Jax with CUDA
RUN pip install --upgrade "jax[cuda11_pip]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Jupyter
RUN pip install jupyterlab ipywidgets

# Other Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 3. Setup Workspace
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
WORKDIR /mnt
COPY . .

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 4. Set Default Command
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
CMD ["bash"]
