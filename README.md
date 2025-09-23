# OLMoCR Endpoint

[![Runpod](https://api.runpod.io/badge/victorftrdba/olmocr-endpoint)](https://console.runpod.io/hub/victorftrdba/olmocr-endpoint)

Um endpoint FastAPI para OCR (Optical Character Recognition) usando o modelo OLMoCR-7B da AllenAI, especializado em reconhecimento de texto em documentos PDF.

## 🚀 Funcionalidades

- **OCR de PDFs**: Extrai texto de documentos PDF usando IA
- **Modelo Avançado**: Utiliza o modelo OLMoCR-7B-0225-preview da AllenAI
- **API REST**: Interface simples via FastAPI
- **Processamento de Imagens**: Converte PDFs em imagens para análise
- **Ancoragem de Texto**: Usa texto de ancoragem para melhor precisão

## 📋 Pré-requisitos

- Python 3.8+
- CUDA (recomendado para melhor performance)
- Docker (opcional)

## 🛠️ Instalação

### Instalação Local

1. Clone o repositório:
```bash
git clone <repository-url>
cd olmocr-endpoint
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute o servidor:
```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

### Instalação com Docker

1. Construa a imagem:
```bash
docker build -t olmocr-endpoint .
```

2. Execute o container:
```bash
docker run -p 8080:8080 --gpus all olmocr-endpoint
```

## 📖 Uso

### Endpoint OCR

**POST** `/ocr`

Envie um arquivo PDF para extrair o texto.

**Parâmetros:**
- `file`: Arquivo PDF (multipart/form-data)

**Exemplo de requisição:**
```bash
curl -X POST "http://localhost:8080/ocr" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@documento.pdf"
```

**Resposta:**
```json
{
  "extracted_text": "Texto extraído do documento PDF..."
}
```

## 🔧 Configuração

O modelo utiliza as seguintes configurações:
- **Modelo**: `allenai/olmOCR-7B-0225-preview`
- **Processor**: `Qwen/Qwen2-VL-7B-Instruct`
- **Temperatura**: 0.8
- **Max Tokens**: 512
- **Resolução de Imagem**: 1024px (dimensão mais longa)

## 📦 Dependências

- `torch`: Framework de deep learning
- `transformers`: Biblioteca de modelos de transformadores
- `Pillow`: Processamento de imagens
- `olmocr`: Biblioteca específica para OCR
- `fastapi`: Framework web
- `uvicorn`: Servidor ASGI

## 🐳 Docker

O projeto inclui um Dockerfile otimizado com:
- Base CUDA 12.2 para suporte a GPU
- Ubuntu 22.04
- Dependências do sistema necessárias
- Configuração automática do servidor

## 📝 Notas

- O modelo é carregado automaticamente na inicialização
- Suporte a GPU quando disponível (CUDA)
- Fallback para CPU quando GPU não está disponível
- Processamento de primeira página do PDF por padrão

## 🤝 Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

## 📄 Licença

Este projeto está sob a licença [especificar licença].
