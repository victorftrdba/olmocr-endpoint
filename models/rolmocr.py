"""
RolmOCR Model Management Module

This module handles the initialization and management of the RolmOCR model
and processor, including memory optimization and error handling.
"""

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


class RolmOCRManager:
    """
    Manages RolmOCR model initialization and operations.
    
    Features:
    - RolmOCR model loading
    - Memory optimization
    - Device detection (CUDA/CPU)
    - Error handling and validation
    """
    
    def __init__(self):
        """Initialize the RolmOCR manager."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.model_name = None
        
    def load_model(self):
        """
        Load the RolmOCR model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
            
        Raises:
            Exception: If RolmOCR fails to load
        """
        if self._try_load_rolmocr():
            return True
            
        # If RolmOCR fails, raise exception
        raise Exception("Failed to load RolmOCR model")
    
    def _try_load_rolmocr(self):
        """
        Load RolmOCR model.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Try loading with bfloat16 first
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                "reducto/RolmOCR",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).eval().to(self.device)
            
            # Use the correct processor for RolmOCR
            self.processor = AutoProcessor.from_pretrained(
                "reducto/RolmOCR",
                trust_remote_code=True
            )
            self.model_name = "RolmOCR"
            return True
            
        except Exception as e:
            print(f"Error loading RolmOCR with bfloat16: {str(e)}")
            try:
                # Fallback: try with float16
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    "reducto/RolmOCR",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                ).eval().to(self.device)
                
                self.processor = AutoProcessor.from_pretrained(
                    "reducto/RolmOCR",
                    trust_remote_code=True
                )
                self.model_name = "RolmOCR"
                return True
                
            except Exception as e2:
                # Log the actual error for debugging
                import traceback
                print(f"Error loading RolmOCR with float16: {str(e2)}")
                print(f"Full traceback: {traceback.format_exc()}")
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
        
        # Validate input parameters
        if temperature < 0.0 or temperature > 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        # Validate image
        if image is None:
            raise ValueError("Image cannot be None")
        
        try:
            # Prepare messages for OCR (using RolmOCR format)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": "Return the plain text representation of this document as if you were reading it naturally.\n",
                        },
                    ],
                }
            ]
            
            # Process input using the processor
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
