"""
RolmOCR Model Management Module

This module handles the initialization and management of the RolmOCR model
and processor, including fallback mechanisms and memory optimization.
"""

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


class RolmOCRManager:
    """
    Manages RolmOCR model initialization and operations.
    
    Features:
    - Automatic model loading with fallback
    - Memory optimization
    - Device detection (CUDA/CPU)
    """
    
    def __init__(self):
        """Initialize the RolmOCR manager."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.model_name = None
        
    def load_model(self):
        """
        Load the RolmOCR model with automatic fallback.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
            
        Raises:
            Exception: If both RolmOCR and olmOCR fail to load
        """
        # Try RolmOCR first
        if self._try_load_rolmocr():
            return True
            
        # Fallback to olmOCR
        if self._try_load_olmocr():
            return True
            
        # If both fail, raise exception
        raise Exception("Failed to load any OCR model")
    
    def _try_load_rolmocr(self):
        """
        Try to load RolmOCR model.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                "reducto/RolmOCR",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            ).eval().to(self.device)
            
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
            self.model_name = "RolmOCR"
            return True
            
        except Exception:
            return False
    
    def _try_load_olmocr(self):
        """
        Try to load olmOCR model as fallback.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                "allenai/olmOCR-7B-0225-preview",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            ).eval().to(self.device)
            
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
            self.model_name = "olmOCR (fallback)"
            return True
            
        except Exception:
            return False
    
    def process_image(self, image, temperature=0.2, max_tokens=4096):
        """
        Process an image with the loaded OCR model.
        
        Args:
            image (PIL.Image): Image to process
            temperature (float): Sampling temperature (0.0-1.0)
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            str: Extracted text from the image
            
        Raises:
            Exception: If model is not loaded or processing fails
        """
        if self.model is None or self.processor is None:
            raise Exception("Model not loaded. Call load_model() first.")
        
        try:
            # Convert image to base64
            from io import BytesIO
            import base64
            
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # Prepare messages for OCR
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
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text_input],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate output
            output = self.model.generate(
                **inputs,
                temperature=temperature,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                do_sample=True,
            )
            
            # Decode output
            prompt_len = inputs["input_ids"].shape[1]
            new_tokens = output[:, prompt_len:]
            text_output = self.processor.tokenizer.batch_decode(
                new_tokens, skip_special_tokens=True
            )
            
            return text_output[0]
            
        except Exception as e:
            raise Exception(f"Failed to process image with {self.model_name}: {str(e)}")
    
    def cleanup_memory(self):
        """
        Clean up GPU memory after processing.
        
        This helps prevent memory leaks and optimizes resource usage.
        """
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information including name and device
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "is_loaded": self.model is not None
        }


# Global model manager instance
model_manager = RolmOCRManager()
