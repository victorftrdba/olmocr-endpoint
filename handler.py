import runpod
import torch
import base64
import requests
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import os
import tempfile
import fitz  # PyMuPDF for PDF processing
from urllib.parse import urlparse
import mimetypes

# Initialize RolmOCR model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "reducto/RolmOCR",
    torch_dtype=torch.bfloat16
).eval().to(device)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

def download_file_from_url(url, max_size_mb=50):
    """
    Download file from URL with size limit
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Check content length
        content_length = response.headers.get('content-length')
        if content_length and int(content_length) > max_size_mb * 1024 * 1024:
            raise ValueError(f"File too large. Maximum size: {max_size_mb}MB")
        
        # Download file
        file_data = BytesIO()
        downloaded_size = 0
        
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                downloaded_size += len(chunk)
                if downloaded_size > max_size_mb * 1024 * 1024:
                    raise ValueError(f"File too large. Maximum size: {max_size_mb}MB")
                file_data.write(chunk)
        
        file_data.seek(0)
        return file_data.getvalue()
        
    except Exception as e:
        raise Exception(f"Failed to download file from URL: {str(e)}")

def pdf_to_images(pdf_data, max_pages=10):
    """
    Convert PDF to images using PyMuPDF
    """
    try:
        # Open PDF from bytes
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        images = []
        
        # Process up to max_pages
        for page_num in range(min(len(pdf_document), max_pages)):
            page = pdf_document[page_num]
            
            # Render page to image
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            image = Image.open(BytesIO(img_data))
            images.append(image)
        
        pdf_document.close()
        return images
        
    except Exception as e:
        raise Exception(f"Failed to convert PDF to images: {str(e)}")

def process_image_with_rolmocr(image, temperature=0.2, max_tokens=4096):
    """
    Process image with RolmOCR model
    """
    try:
        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Prepare messages for RolmOCR
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                    {
                        "type": "text",
                        "text": "Return the plain text representation of this document as if you were reading it naturally.\n",
                    },
                ],
            }
        ]
        
        # Process input
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(
            text=[text_input],
            images=[image],
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
        
        return text_output[0]
        
    except Exception as e:
        raise Exception(f"Failed to process image with RolmOCR: {str(e)}")

def handler(job):
    """
    RunPod serverless handler for OCR processing with RolmOCR
    """
    try:
        # Get job input
        job_input = job["input"]
        
        # Get environment variables with defaults
        max_tokens = int(os.getenv("MAX_TOKENS", "4096"))
        temperature = float(os.getenv("TEMPERATURE", "0.2"))
        max_pages = int(os.getenv("MAX_PAGES", "10"))
        max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
        
        # Handle URL input
        if "url" in job_input:
            url = job_input["url"]
            
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return {"error": "Invalid URL provided"}
            
            # Download file from URL
            file_data = download_file_from_url(url, max_file_size_mb)
            
            # Determine file type from URL or content
            file_extension = os.path.splitext(parsed_url.path)[1].lower()
            if not file_extension:
                # Try to detect from content type
                try:
                    response = requests.head(url, timeout=10)
                    content_type = response.headers.get('content-type', '')
                    if 'pdf' in content_type:
                        file_extension = '.pdf'
                    elif 'image' in content_type:
                        file_extension = '.png'
                except:
                    file_extension = '.pdf'  # Default to PDF
            
        # Handle direct file input (backward compatibility)
        elif "file" in job_input:
            if isinstance(job_input["file"], str):
                file_data = base64.b64decode(job_input["file"])
                file_extension = job_input.get("file_extension", "pdf")
            else:
                file_data = job_input["file"]
                file_extension = job_input.get("file_extension", "pdf")
        else:
            return {"error": "No URL or file provided in input"}
        
        # Process file based on type
        extracted_texts = []
        
        if file_extension.lower() in ['.pdf']:
            # Convert PDF to images
            images = pdf_to_images(file_data, max_pages)
            
            # Process each page
            for i, image in enumerate(images):
                try:
                    text = process_image_with_rolmocr(image, temperature, max_tokens)
                    extracted_texts.append({
                        "page": i + 1,
                        "text": text
                    })
                except Exception as e:
                    extracted_texts.append({
                        "page": i + 1,
                        "text": f"Error processing page: {str(e)}"
                    })
        
        elif file_extension.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
            # Process single image
            try:
                image = Image.open(BytesIO(file_data))
                text = process_image_with_rolmocr(image, temperature, max_tokens)
                extracted_texts.append({
                    "page": 1,
                    "text": text
                })
            except Exception as e:
                return {"error": f"Failed to process image: {str(e)}"}
        
        else:
            return {"error": f"Unsupported file type: {file_extension}"}
        
        # Combine all extracted text
        combined_text = "\n\n".join([page["text"] for page in extracted_texts])
        
        return {
            "extracted_text": combined_text,
            "pages": extracted_texts,
            "total_pages": len(extracted_texts),
            "status": "success"
        }
                
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

# Start the RunPod serverless handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
