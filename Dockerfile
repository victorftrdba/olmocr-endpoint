# Usa Python 3.11 otimizado
FROM python:3.11-slim

WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

# Copia requirements e instala dependências
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia o handler
COPY handler.py .

# Define o comando de inicialização
CMD ["python", "-u", "handler.py"]
