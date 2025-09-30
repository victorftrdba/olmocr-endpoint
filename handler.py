"""
RolmOCR RunPod Serverless Handler

This handler implements OCR functionality using the RolmOCR model from Reducto AI.
It processes image URLs and returns extracted text following the RolmOCR documentation.

Based on:
- RunPod Serverless Handler Functions: https://docs.runpod.io/serverless/workers/handler-functions
- RolmOCR Model: https://huggingface.co/reducto/RolmOCR
"""

import runpod
import base64
import requests
from PIL import Image
import io
import logging
from typing import Dict, Any, Optional
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model initialization
model = None
processor = None
device = None

def initialize_model():
    """
    Initialize the RolmOCR model and processor.
    This function loads the model outside the handler for better performance.
    """
    global model, processor, device
    
    try:
        logger.info("Initializing RolmOCR model...")
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Load model and processor
        model_name = "reducto/RolmOCR"
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("RolmOCR model initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise e

def download_image_from_url(url: str, timeout: int = 30) -> Image.Image:
    """
    Download image from URL and return PIL Image object.
    
    Args:
        url (str): Image URL
        timeout (int): Request timeout in seconds
        
    Returns:
        PIL.Image: Downloaded image
        
    Raises:
        Exception: If download or processing fails
    """
    try:
        logger.info(f"Downloading image from: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, timeout=timeout, headers=headers, stream=True)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if not content_type.startswith('image/'):
            raise ValueError(f"URL does not point to an image. Content-Type: {content_type}")
        
        # Load image
        image_data = io.BytesIO(response.content)
        image = Image.open(image_data)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        logger.info(f"Image downloaded successfully: {image.size}")
        return image
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download image: {str(e)}")
    except Exception as e:
        raise Exception(f"Failed to process image: {str(e)}")

def encode_image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 string.
    
    Args:
        image (PIL.Image): Image to encode
        
    Returns:
        str: Base64 encoded image
    """
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')

def process_image_with_rolmocr(image: Image.Image) -> str:
    """
    Process image using RolmOCR model to extract text.
    
    Args:
        image (PIL.Image): Image to process
        
    Returns:
        str: Extracted text
    """
    global model, processor, device
    
    try:
        if model is None or processor is None:
            raise Exception("Model not initialized")
        
        logger.info("Processing image with RolmOCR...")
        
        # Prepare the prompt as specified in RolmOCR documentation
        prompt = "Return the plain text representation of this document as if you were reading it naturally.\n"
        
        # Prepare messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Apply chat template and process
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = processor.process_vision_info(messages)
        
        # Prepare inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.2,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        logger.info("OCR processing completed successfully")
        return output_text.strip()
        
    except Exception as e:
        logger.error(f"OCR processing failed: {str(e)}")
        raise Exception(f"Failed to process image with RolmOCR: {str(e)}")

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod serverless handler for OCR processing.
    
    This function processes incoming requests containing image URLs
    and returns extracted text using the RolmOCR model.
    
    Args:
        job (dict): RunPod job containing input data with 'image_url' field
        
    Returns:
        dict: Processing result with extracted text and metadata
    """
    try:
        # Get job input
        job_input = job.get("input", {})
        
        if not job_input:
            return {
                "error": "No input provided. Please provide 'image_url' in the input.",
                "status": "error"
            }
        
        # Validate required input
        image_url = job_input.get("image_url")
        if not image_url:
            return {
                "error": "Missing required field 'image_url'. Please provide a valid image URL.",
                "status": "error"
            }
        
        # Validate URL format
        if not isinstance(image_url, str) or not image_url.startswith(('http://', 'https://')):
            return {
                "error": "Invalid image_url format. Please provide a valid HTTP/HTTPS URL.",
                "status": "error"
            }
        
        # Initialize model if not already done
        if model is None:
            try:
                initialize_model()
            except Exception as model_error:
                return {
                    "error": f"Model initialization failed: {str(model_error)}",
                    "status": "error"
                }
        
        # Download and process image
        try:
            image = download_image_from_url(image_url)
            extracted_text = process_image_with_rolmocr(image)
            
            return {
                "status": "success",
                "extracted_text": extracted_text,
                "image_url": image_url,
                "image_size": image.size,
                "model": "reducto/RolmOCR"
            }
            
        except Exception as processing_error:
            return {
                "error": f"Image processing failed: {str(processing_error)}",
                "status": "error",
                "image_url": image_url
            }
        
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {
            "error": f"Unexpected error: {str(e)}",
            "status": "error"
        }

# Initialize model when module is imported (outside handler for better performance)
if __name__ == "__main__":
    try:
        initialize_model()
        logger.info("Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})
    except Exception as e:
        logger.error(f"Failed to start handler: {str(e)}")
        raise