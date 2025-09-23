import runpod
import torch
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text
import os
import tempfile

# Initialize model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "allenai/olmOCR-7B-0225-preview",
    torch_dtype=torch.bfloat16
).eval().to(device)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

def handler(job):
    """
    RunPod serverless handler for OCR processing
    """
    try:
        # Get job input
        job_input = job["input"]
        
        # Get environment variables with defaults
        max_tokens = int(os.getenv("MAX_TOKENS", "512"))
        temperature = float(os.getenv("TEMPERATURE", "0.8"))
        target_length = int(os.getenv("TARGET_LENGTH", "4000"))
        pdf_engine = os.getenv("PDF_ENGINE", "pdfreport")
        
        # Handle file input
        if "file" in job_input:
            # If file is provided as base64 string
            if isinstance(job_input["file"], str):
                file_data = base64.b64decode(job_input["file"])
                file_extension = job_input.get("file_extension", "pdf")
            else:
                # If file is provided as binary data
                file_data = job_input["file"]
                file_extension = job_input.get("file_extension", "pdf")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as tmp_file:
                tmp_file.write(file_data)
                pdf_path = tmp_file.name
        else:
            return {"error": "No file provided in input"}
        
        try:
            # Process PDF
            image_base64 = render_pdf_to_base64png(pdf_path, 1, target_longest_image_dim=1024)
            main_image = Image.open(BytesIO(base64.b64decode(image_base64)))
            
            # Get anchor text
            anchor_text = get_anchor_text(pdf_path, 1, pdf_engine=pdf_engine, target_length=target_length)
            prompt = build_finetuning_prompt(anchor_text)
            
            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ]
            
            # Process input
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = processor(
                text=[text_input],
                images=[main_image],
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate output
            output = model.generate(
                **inputs,
                temperature=temperature,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                do_sample=True,
            )
            
            # Decode output
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = output[:, prompt_len:]
            text_output = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            
            return {
                "extracted_text": text_output[0],
                "anchor_text": anchor_text,
                "status": "success"
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
                
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

# Start the RunPod serverless handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
