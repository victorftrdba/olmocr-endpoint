FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# --- Dependências do sistema ---
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    poppler-utils \
    fonts-dejavu \
    gsfonts \
    && rm -rf /var/lib/apt/lists/*

# --- Atualizar pip ---
RUN pip install --upgrade pip

# --- Instala PyTorch direto do repositório oficial (GPU compatível com CUDA 12.1) ---
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cu121

# --- Instala olmOCR e dependências ---
RUN pip install olmocr \
    transformers \
    pillow \
    fastapi \
    uvicorn

# --- Requisitos adicionais do projeto ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || true

# --- Copiar código ---
COPY . .

# --- Porta padrão para FastAPI ---
EXPOSE 8080

# --- Comando de inicialização ---
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
