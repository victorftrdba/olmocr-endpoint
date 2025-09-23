from fastapi import FastAPI, UploadFile, File
import torch, base64
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "allenai/olmOCR-7B-0225-preview",
    torch_dtype=torch.bfloat16
).eval().to(device)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    pdf_path = f"/tmp/{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    image_base64 = render_pdf_to_base64png(pdf_path, 1, target_longest_image_dim=1024)
    main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

    anchor_text = get_anchor_text(pdf_path, 1, pdf_engine="pdfreport", target_length=4000)
    prompt = build_finetuning_prompt(anchor_text)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }
    ]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text_input],
        images=[main_image],
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        temperature=0.8,
        max_new_tokens=512,
        num_return_sequences=1,
        do_sample=True,
    )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_len:]
    text_output = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    return {"extracted_text": text_output[0]}
