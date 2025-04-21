# Base image: NVIDIA CUDA 11.8 + cuDNN 8 + Ubuntu 22.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Install required system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tzdata \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    python3-tk \
    build-essential \
    git wget curl vim tmux nano ffmpeg \
    libgl1-mesa-glx libglib2.0-0 x11-apps \
    && ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Setup Python3 
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Setup cuda 
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PATH /usr/local/cuda/bin:${PATH}

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Install numpy
RUN pip install --no-cache-dir numpy==1.24.4

# Install TensorFlow & Waymo Open Dataset
RUN pip install tensorflow==2.12.0
RUN pip install waymo-open-dataset-tf-2-12-0==1.6.4

# JAX for CUDA 11.8
RUN pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install waymax (Waymo behavior sim)
RUN pip install git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax

# Google cloud tools (optional)
RUN pip install gsutil gcsfs

# Jupyter for dev
RUN pip install jupyterlab

# Copy and install local requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code into the container
COPY . .

# Set working directory
WORKDIR /mnt

# Expose the Jupyter Notebook port
EXPOSE 9999

# Default command to run the project
# CMD ["/bin/bash"]
CMD ["bash", "-c", "cd /mnt && exec bash"]
