# RolmOCR Endpoint

[![Runpod](https://api.runpod.io/badge/victorftrdba/rolmocr-endpoint)](https://console.runpod.io/hub/victorftrdba/rolmocr-endpoint)

Um endpoint RunPod Serverless para OCR (Optical Character Recognition) usando o modelo RolmOCR-7B da Reducto AI, uma versão otimizada e mais rápida do olmOCR original.

## 🚀 Funcionalidades

- **OCR de PDFs e Imagens**: Extrai texto de documentos PDF e imagens usando IA
- **Modelo Otimizado**: Utiliza o modelo RolmOCR-7B (mais rápido e eficiente)
- **Suporte a URLs**: Processa arquivos diretamente de URLs
- **Múltiplas Páginas**: Processa PDFs com múltiplas páginas
- **Formatos Suportados**: PDF, PNG, JPG, JPEG, GIF, BMP, TIFF
- **API Serverless**: Interface via RunPod.io
- **Processamento Inteligente**: Converte PDFs em imagens para análise otimizada

## 📋 Pré-requisitos

- Python 3.11+
- CUDA 12.2+ (recomendado para melhor performance)
- RunPod.io account
- Docker (para build local)

## 🛠️ Deploy no RunPod.io

### Opção 1: Deploy via GitHub (Recomendado)

1. Faça push do código para o GitHub
2. No RunPod.io, crie um novo "Serverless Endpoint"
3. Conecte com seu repositório GitHub
4. Configure as variáveis de ambiente (opcional):
   - `MAX_TOKENS=4096`
   - `TEMPERATURE=0.2`
   - `MAX_PAGES=10`
   - `MAX_FILE_SIZE_MB=50`

### Opção 2: Deploy via Docker Hub

1. Construa a imagem:
```bash
docker build -t seu-usuario/rolmocr-endpoint .
```

2. Faça push para Docker Hub:
```bash
docker push seu-usuario/rolmocr-endpoint
```

3. No RunPod.io, use a imagem: `seu-usuario/rolmocr-endpoint:latest`

## 📖 Uso

### API Endpoint

**POST** `https://seu-endpoint.runpod.net/v2/LOCAL/runsync`

### Opção 1: Processar arquivo via URL (Recomendado)

**Exemplo de requisição:**
```python
import requests

data = {
    "input": {
        "url": "https://example.com/document.pdf",
        "max_pages": 10,
        "temperature": 0.2,
        "max_tokens": 4096
    }
}

response = requests.post("https://seu-endpoint.runpod.net/v2/LOCAL/runsync", json=data)
result = response.json()
```

### Opção 2: Processar arquivo via Base64

**Exemplo de requisição:**
```python
import base64
import requests

# Converter arquivo para base64
with open("document.pdf", "rb") as f:
    file_data = base64.b64encode(f.read()).decode('utf-8')

data = {
    "input": {
        "file": file_data,
        "file_extension": "pdf"
    }
}

response = requests.post("https://seu-endpoint.runpod.net/v2/LOCAL/runsync", json=data)
result = response.json()
```

### Resposta da API

```json
{
  "extracted_text": "Texto completo extraído do documento...",
  "pages": [
    {
      "page": 1,
      "text": "Texto da página 1..."
    },
    {
      "page": 2,
      "text": "Texto da página 2..."
    }
  ],
  "total_pages": 2,
  "status": "success"
}
```

## 🔧 Configuração

O modelo utiliza as seguintes configurações:
- **Modelo**: `reducto/RolmOCR-7b`
- **Processor**: `Qwen/Qwen2.5-VL-7B-Instruct`
- **Temperatura**: 0.2 (padrão)
- **Max Tokens**: 4096 (padrão)
- **Max Páginas**: 10 (padrão)
- **Tamanho Máximo**: 50MB (padrão)

### Variáveis de Ambiente

- `MAX_TOKENS`: Máximo de tokens na resposta (padrão: 4096)
- `TEMPERATURE`: Criatividade da resposta 0.0-1.0 (padrão: 0.2)
- `MAX_PAGES`: Máximo de páginas a processar (padrão: 10)
- `MAX_FILE_SIZE_MB`: Tamanho máximo do arquivo em MB (padrão: 50)

## 📦 Dependências

- `torch`: Framework de deep learning
- `transformers`: Biblioteca de modelos de transformadores
- `Pillow`: Processamento de imagens
- `requests`: Requisições HTTP
- `PyMuPDF`: Processamento de PDFs
- `runpod`: SDK do RunPod.io

## 🐳 Docker

O projeto inclui um Dockerfile otimizado com:
- Base CUDA 12.2 para suporte a GPU
- Ubuntu 22.04 com Python 3.11
- Dependências do sistema necessárias
- Download dos modelos durante o build
- Configuração automática do servidor

## 📝 Notas

- O modelo RolmOCR é mais rápido e eficiente que o olmOCR original
- Suporte a GPU quando disponível (CUDA)
- Fallback para CPU quando GPU não está disponível
- Processamento de múltiplas páginas de PDF
- Suporte a URLs para download automático de arquivos
- Compatibilidade com formato base64 para arquivos locais

## 🚀 Vantagens do RolmOCR

- **Mais Rápido**: Até 2x mais rápido que olmOCR original
- **Menos Memória**: Usa menos VRAM durante o processamento
- **Melhor Qualidade**: Mantém a mesma qualidade de OCR
- **Sem Metadata**: Não precisa de metadata de PDF (mais simples)
- **Robustez**: Treinado com dados rotacionados para melhor precisão

## 📊 Formatos Suportados

### Documentos
- **PDF**: Múltiplas páginas (até 10 por padrão)

### Imagens
- **PNG**: Imagens PNG
- **JPG/JPEG**: Imagens JPEG
- **GIF**: Imagens GIF
- **BMP**: Imagens BMP
- **TIFF**: Imagens TIFF

## 🔍 Exemplos de Uso

### Processar PDF via URL
```python
import requests

data = {
    "input": {
        "url": "https://example.com/relatorio.pdf",
        "max_pages": 5
    }
}

response = requests.post("https://seu-endpoint.runpod.net/v2/LOCAL/runsync", json=data)
print(response.json()["extracted_text"])
```

### Processar Imagem via URL
```python
data = {
    "input": {
        "url": "https://example.com/nota_fiscal.jpg"
    }
}

response = requests.post("https://seu-endpoint.runpod.net/v2/LOCAL/runsync", json=data)
print(response.json()["extracted_text"])
```

## 🤝 Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

## 📄 Licença

Este projeto está sob a licença Apache 2.0.
