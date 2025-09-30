# Base: Python 3.11 + CUDA 12.2 for optimal GPU performance
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    git \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    python3-pip \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    pkg-config \
    libjpeg-dev \
    zlib1g-dev \
    fonts-dejavu \
    gsfonts \
    wget \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip and install basic dependencies
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.2 support
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu122

# Install other Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directory
RUN mkdir -p /app/.cache/huggingface

# Copy application code
COPY . .

# Create a non-root user for security
RUN useradd -m -u 1000 runpod && chown -R runpod:runpod /app
USER runpod

# Expose port (optional for RunPod Serverless)
EXPOSE 8080

# Start the RunPod serverless handler
CMD ["python", "handler.py"]