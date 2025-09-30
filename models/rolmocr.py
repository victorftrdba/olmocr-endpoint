"""
RolmOCR Model Management Module

This module handles the initialization and management of the RolmOCR model
and processor, including memory optimization and error handling.
"""

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image


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
        print("Starting RolmOCR model loading...")
        if self._try_load_rolmocr():
            print(f"Successfully loaded {self.model_name} model on {self.device}")
            return True
            
        # If RolmOCR fails, raise exception with detailed error message
        raise Exception("Failed to load RolmOCR model. This may be due to model architecture mismatches or insufficient resources. Check the logs above for detailed error information.")
    
    def _try_load_rolmocr(self):
        """
        Load RolmOCR model.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Try loading with bfloat16 first
            print("Attempting to load RolmOCR with bfloat16...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                "reducto/RolmOCR",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
                device_map="auto" if torch.cuda.is_available() else None
            ).eval()
            
            # Use the correct processor for RolmOCR
            self.processor = AutoProcessor.from_pretrained(
                "reducto/RolmOCR",
                trust_remote_code=True
            )
            self.model_name = "RolmOCR"
            return True
            
        except Exception as e:
            print(f"Error loading RolmOCR with bfloat16: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            try:
                # Fallback: try with float16
                print("Attempting to load RolmOCR with float16...")
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    "reducto/RolmOCR",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    ignore_mismatched_sizes=True,
                    device_map="auto" if torch.cuda.is_available() else None
                ).eval()
                
                self.processor = AutoProcessor.from_pretrained(
                    "reducto/RolmOCR",
                    trust_remote_code=True
                )
                self.model_name = "RolmOCR"
                return True
                
            except Exception as e2:
                print(f"Error loading RolmOCR with float16: {str(e2)}")
                print(f"Error type: {type(e2).__name__}")
                try:
                    # Final fallback: try with default dtype and ignore mismatched sizes
                    print("Attempting final fallback with default dtype...")
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        "reducto/RolmOCR",
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        ignore_mismatched_sizes=True,
                        device_map="auto" if torch.cuda.is_available() else None
                    ).eval()
                    
                    self.processor = AutoProcessor.from_pretrained(
                        "reducto/RolmOCR",
                        trust_remote_code=True
                    )
                    self.model_name = "RolmOCR"
                    print("Successfully loaded RolmOCR with default dtype")
                    return True
                    
                except Exception as e3:
                    # Log the actual error for debugging
                    import traceback
                    print(f"Error loading RolmOCR with default dtype: {str(e3)}")
                    print(f"Error type: {type(e3).__name__}")
                    print(f"Full traceback: {traceback.format_exc()}")
                    return False
    
    def _preprocess_image(self, image):
        """
        Preprocess image to ensure compatibility with RolmOCR model.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Preprocessed image
        """
        try:
            # Convert to RGB if necessary (handles RGBA, L, P modes)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Ensure image is not too large (common cause of token/feature mismatch)
            max_size = 1024  # Reasonable size for vision models
            if max(image.size) > max_size:
                # Calculate new size maintaining aspect ratio
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Ensure minimum size (very small images can cause issues)
            min_size = 32
            if min(image.size) < min_size:
                # Calculate new size maintaining aspect ratio
                ratio = min_size / min(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            # If preprocessing fails, return original image
            print(f"Warning: Image preprocessing failed: {str(e)}")
            return image
    
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
        
        # Preprocess image to ensure compatibility
        image = self._preprocess_image(image)
        
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
            
            # Process inputs with error handling for token/feature mismatch
            try:
                inputs = self.processor(
                    text=[text_input],
                    images=[image],
                    padding=True,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            except Exception as proc_error:
                # If processor fails, try with different settings
                print(f"Processor error: {str(proc_error)}")
                print(f"Image size: {image.size}, mode: {image.mode}")
                
                # Try with different processor settings
                try:
                    inputs = self.processor(
                        text=[text_input],
                        images=[image],
                        padding=False,  # Try without padding
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                except Exception as proc_error2:
                    raise Exception(f"Failed to process image with processor: {str(proc_error2)}")
            
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
