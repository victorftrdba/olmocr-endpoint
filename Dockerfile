# Base: Python 3.11 + CUDA 12.2 (melhor abordagem)
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

WORKDIR /app

# --- Configurar timezone para evitar prompts interativos ---
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# --- Instalar Python 3.11 e dependências do sistema ---
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
    poppler-utils \
    pkg-config \
    libjpeg-dev \
    zlib1g-dev \
    fonts-dejavu \
    gsfonts \
    wget \
    curl \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# --- Configurar Python 3.11 como padrão ---
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# --- Atualizar pip e instalar dependências básicas ---
RUN pip install --upgrade pip setuptools wheel

# --- Copiar requirements.txt primeiro para cache de dependências ---
COPY requirements.txt .

# --- Instalar dependências Python ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Instalar PyTorch compatível CUDA 12.2 ---
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu122

# --- Copiar handler.py ---
COPY handler.py .

# --- Configurar variáveis de ambiente ---
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# --- Porta padrão (opcional, RunPod Serverless não exige) ---
EXPOSE 8080

# --- Comando de inicialização RunPod Serverless ---
CMD ["python", "handler.py"]
