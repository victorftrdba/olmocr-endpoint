# Base: imagem oficial do olmOCR (já contém Python 3.11, PyTorch com CUDA, olmOCR e dependências)
FROM alleninstituteforai/olmocr:latest

WORKDIR /app

# --- Copiar handler e requisitos adicionais ---
COPY handler.py .
COPY requirements.txt .

# --- Instalar libs extras necessárias para o handler ---
RUN pip install --no-cache-dir -r requirements.txt

# --- Porta padrão (opcional, Serverless não precisa expor manualmente) ---
EXPOSE 8080

# --- Comando de inicialização do RunPod Serverless ---
CMD ["python", "handler.py"]
