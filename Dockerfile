# Base: CUDA 12.2 + cuDNN 8 runtime
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# --- Dependências do sistema necessárias para olmOCR e PDF ---
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    python3-venv \
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
    && rm -rf /var/lib/apt/lists/*

# --- Atualizar pip e instalar dependências básicas ---
RUN pip install --upgrade pip setuptools wheel

# --- Copiar requirements.txt primeiro para cache de dependências ---
COPY requirements.txt .

# --- Instalar dependências Python ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Instalar PyTorch compatível CUDA 12.2 ---
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu122

# --- Instalar olmOCR do GitHub ---
RUN pip install git+https://github.com/allenai/olmocr.git

# --- Copiar handler.py ---
COPY handler.py .

# --- Configurar variáveis de ambiente ---
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# --- Porta padrão (opcional, RunPod Serverless não exige) ---
EXPOSE 8080

# --- Comando de inicialização RunPod Serverless ---
CMD ["python", "handler.py"]
