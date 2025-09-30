"""
RolmOCR Model Management Module

This module handles the initialization and management of the RolmOCR model
and processor, including memory optimization and error handling.
"""

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from config import Config


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
            
            # Fix potential image token ID mismatch issues
            self._fix_image_token_config()
            
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
                
                # Fix potential image token ID mismatch issues
                self._fix_image_token_config()
                
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
                    
                    # Fix potential image token ID mismatch issues
                    self._fix_image_token_config()
                    
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
    
    def _fix_image_token_config(self):
        """
        Fix potential image token ID mismatch issues in the model configuration.
        
        This addresses the common issue where the model's image_token_id doesn't
        match the actual token ID used by the tokenizer.
        """
        try:
            if hasattr(self.model, 'config') and hasattr(self.processor, 'tokenizer'):
                print("Checking model configuration for potential fixes...")
                
                # Check max_prompt_length to ensure it's sufficient
                if hasattr(self.model.config, 'max_prompt_length'):
                    if self.model.config.max_prompt_length < 4096:
                        print(f"Increasing max_prompt_length from {self.model.config.max_prompt_length} to 4096")
                        self.model.config.max_prompt_length = 4096
                
                # Check if there's an image token ID mismatch
                if hasattr(self.model.config, 'image_token_id'):
                    print(f"Model config image_token_id: {self.model.config.image_token_id}")
                    
                    # Get the actual image token ID from the tokenizer
                    image_token_id = None
                    
                    # Try different possible image token formats
                    possible_tokens = ['<image>', '<IMG_CONTEXT>', '<img>', '<IMAGE>', '<|image_pad|>']
                    for token in possible_tokens:
                        try:
                            token_id = self.processor.tokenizer.convert_tokens_to_ids(token)
                            if token_id != self.processor.tokenizer.unk_token_id:
                                image_token_id = token_id
                                print(f"Found image token '{token}' with ID: {token_id}")
                                break
                        except Exception as token_error:
                            print(f"Error checking token '{token}': {str(token_error)}")
                            continue
                    
                    # Update model config if we found a mismatch
                    if image_token_id is not None and image_token_id != self.model.config.image_token_id:
                        print(f"Fixing image token ID mismatch: {self.model.config.image_token_id} -> {image_token_id}")
                        self.model.config.image_token_id = image_token_id
                    else:
                        print("No image token ID mismatch found")
                else:
                    print("Model config does not have image_token_id attribute")
                    
                print("Model configuration check completed")
                            
        except Exception as e:
            print(f"Warning: Could not fix image token config: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
    
    def _preprocess_image(self, image):
        """
        Preprocess image to ensure compatibility with RolmOCR model.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Preprocessed image
        """
        try:
            print(f"Original image: {image.size}, mode: {image.mode}")
            
            # Convert to RGB if necessary (handles RGBA, L, P modes)
            if image.mode != 'RGB':
                image = image.convert('RGB')
                print(f"Converted to RGB: {image.size}, mode: {image.mode}")
            
            # For RolmOCR, use more conservative resizing to avoid token/feature mismatches
            max_size = getattr(Config, 'MAX_IMAGE_SIZE', 2048)
            min_size = getattr(Config, 'MIN_IMAGE_SIZE', 64)
            
            # Only resize if absolutely necessary
            width, height = image.size
            
            # Check if image is too large
            if max(width, height) > max_size:
                ratio = max_size / max(width, height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"Resized large image to: {new_width}x{new_height}")
            
            # Check if image is too small
            elif min(width, height) < min_size:
                ratio = min_size / min(width, height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"Upscaled small image to: {new_width}x{new_height}")
            
            # Don't force dimensions to be multiples of 16 - this can cause issues
            # Just ensure the image is reasonable size
            final_width, final_height = image.size
            print(f"Final image size: {final_width}x{final_height}")
            
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
            # Use a much simpler approach for RolmOCR
            # RolmOCR works better with direct text prompts rather than complex chat templates
            prompt = "Extract all text from this image. Return only the plain text content without any formatting or explanations."
            
            # Process inputs with multiple fallback strategies
            inputs = None
            last_error = None
            
            # Strategy 1: Try with simple text and image
            try:
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                print(f"Successfully processed with simple prompt")
                
            except Exception as e1:
                last_error = e1
                print(f"Simple processing failed: {str(e1)}")
                
                # Strategy 2: Try with chat template
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                    
                    text_input = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    
                    inputs = self.processor(
                        text=text_input,
                        images=image,
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    print(f"Successfully processed with chat template")
                    
                except Exception as e2:
                    last_error = e2
                    print(f"Chat template processing failed: {str(e2)}")
                    
                    # Strategy 3: Try with minimal settings
                    try:
                        inputs = self.processor(
                            text="What text do you see in this image?",
                            images=image,
                            return_tensors="pt"
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        print(f"Successfully processed with minimal prompt")
                        
                    except Exception as e3:
                        last_error = e3
                        print(f"Minimal processing failed: {str(e3)}")
                        
                        # Strategy 4: Try without any text
                        try:
                            inputs = self.processor(
                                images=image,
                                return_tensors="pt"
                            )
                            inputs = {k: v.to(self.device) for k, v in inputs.items()}
                            print(f"Successfully processed with image only")
                            
                        except Exception as e4:
                            raise Exception(f"All processing strategies failed. Last error: {str(e4)}")
            
            if inputs is None:
                raise Exception(f"Failed to process image: {str(last_error)}")
            
            # Generate output with conservative settings to avoid token/feature mismatch
            try:
                output = self.model.generate(
                    **inputs,
                    temperature=temperature,
                    max_new_tokens=min(max_tokens, 2048),  # Limit max tokens to avoid issues
                    num_return_sequences=1,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            except Exception as gen_error:
                print(f"Generation error: {str(gen_error)}")
                # Try with even more conservative settings
                output = self.model.generate(
                    **inputs,
                    temperature=0.1,  # Lower temperature
                    max_new_tokens=1024,  # Even lower max tokens
                    num_return_sequences=1,
                    do_sample=False,  # Greedy decoding
                    pad_token_id=self.processor.tokenizer.eos_token_id,
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
