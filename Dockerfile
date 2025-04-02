# Base image: NVIDIA CUDA 11.8 + cuDNN 8 + Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install required system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    build-essential \
    git wget curl vim tmux nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Setup Python3 
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Setup cuda 
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PATH /usr/local/cuda/bin:${PATH}

# Upgrade pip, setuptools, and wheel
RUN pip3 install --upgrade pip setuptools wheel

# Install JAX GPU version compatible with CUDA 11.8
RUN pip3 install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install PyTorch (CUDA version) and TensorFlow
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install tensorflow waymo-open-dataset-tf-2-12-0==1.6.4

# Install Jupyter Notebook
RUN pip3 install jupyter

# Install additional dependencies from requirements.txt
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project code into the container
COPY . .

# Set working directory
WORKDIR /workspace

# Expose the Jupyter Notebook port
EXPOSE 9999

# Default command to run the project
# CMD ["/bin/bash"]
CMD ["bash", "-c", "cd /mnt && exec bash"]
