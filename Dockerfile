# Base: CUDA 12.2 + cuDNN 8 runtime
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# --- Dependências do sistema necessárias para olmOCR e PDF ---
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
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
    && rm -rf /var/lib/apt/lists/*

# --- Atualizar pip ---
RUN pip install --upgrade pip

# --- Instalar PyTorch compatível CUDA 12.2 ---
RUN pip install torch --index-url https://download.pytorch.org/whl/cu122

# --- Instalar olmOCR do GitHub + libs extras ---
RUN pip install git+https://github.com/allenai/olmocr.git \
    transformers \
    Pillow \
    runpod

# --- Copiar handler.py ---
COPY handler.py .

# --- Porta padrão (opcional, RunPod Serverless não exige) ---
EXPOSE 8080

# --- Comando de inicialização RunPod Serverless ---
CMD ["python", "handler.py"]
