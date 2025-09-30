# RolmOCR RunPod Serverless Endpoint

Este projeto implementa um endpoint serverless no RunPod que utiliza o modelo RolmOCR da Reducto AI para extrair texto de imagens via URL.

## Sobre o RolmOCR

O RolmOCR é uma versão otimizada do olmOCR original, oferecendo:
- **Maior velocidade**: Processamento mais rápido que o modelo original
- **Menor uso de memória**: Redução significativa no uso de VRAM
- **Melhor precisão**: Mantém a qualidade de extração de texto
- **Baseado no Qwen2.5-VL-7B**: Modelo mais recente e eficiente

## Funcionalidades

- ✅ Processamento de imagens via URL
- ✅ Extração de texto usando RolmOCR
- ✅ Suporte a múltiplos formatos de imagem (PNG, JPG, JPEG, GIF, BMP, TIFF)
- ✅ Tratamento de erros robusto
- ✅ Logging detalhado
- ✅ Otimizado para RunPod Serverless

## Como Usar

### Input Esperado

```json
{
    "input": {
        "image_url": "https://example.com/image.png"
    }
}
```

### Output Retornado

```json
{
    "status": "success",
    "extracted_text": "Texto extraído da imagem...",
    "image_url": "https://example.com/image.png",
    "image_size": [width, height],
    "model": "reducto/RolmOCR"
}
```

### Exemplo de Erro

```json
{
    "error": "Descrição do erro",
    "status": "error",
    "image_url": "https://example.com/image.png"
}
```

## Teste Local

Para testar localmente:

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

2. Execute o handler:
```bash
python handler.py
```

3. Ou teste com input específico:
```bash
python handler.py --test_input '{"input": {"image_url": "https://example.com/image.png"}}'
```

## Deploy no RunPod

1. Construa a imagem Docker:
```bash
docker build -t rolmoocr-endpoint .
```

2. Faça push para um registry (Docker Hub, etc.)

3. Crie um endpoint serverless no RunPod usando a imagem

## Configuração

O endpoint está configurado com os seguintes parâmetros padrão:
- **Temperature**: 0.2 (para resultados mais determinísticos)
- **Max Tokens**: 4096
- **Timeout de Download**: 30 segundos
- **Device**: CUDA (se disponível) ou CPU

## Limitações

- URLs devem apontar para imagens válidas
- Tamanho máximo de resposta: 10MB (endpoint `/run`) ou 20MB (endpoint `/runsync`)
- O modelo pode ocasionalmente ter alucinações ou perder conteúdo
- Não retorna coordenadas de bounding boxes (apenas texto)

## Dependências Principais

- `runpod`: SDK do RunPod para serverless
- `transformers`: Biblioteca Hugging Face para modelos
- `torch`: PyTorch para inferência
- `Pillow`: Processamento de imagens
- `requests`: Download de imagens

## Referências

- [Documentação RunPod Handler Functions](https://docs.runpod.io/serverless/workers/handler-functions)
- [Modelo RolmOCR no Hugging Face](https://huggingface.co/reducto/RolmOCR)
- [olmOCR Original](https://huggingface.co/allenai/olmOCR-mix-0225)