import runpod
import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from PIL import Image
import requests
from io import BytesIO

# ---------------------------------------------------
# 🔹 Carrega o modelo uma única vez (fora do handler)
# ---------------------------------------------------
MODEL_NAME = "reducto/RolmOCR"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# ---------------------------------------------------
# 🔹 Função principal do worker
# ---------------------------------------------------
def handler(job):
    job_input = job.get("input", {})
    image_url = job_input.get("image_url")

    if not image_url:
        return {"error": "Missing 'image_url' in input."}

    try:
        # Baixa a imagem e converte
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")

        # Pré-processamento e inferência
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        with torch.inference_mode():
            generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return {"text": text.strip()}

    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------
# 🔹 Inicializa o servidor RunPod
# ---------------------------------------------------
runpod.serverless.start({"handler": handler})
