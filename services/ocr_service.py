"""
OCR Service Module

This module provides the main OCR processing service that coordinates
file processing, model operations, and result formatting.
"""

import os
import base64
from io import BytesIO
from PIL import Image

from models.rolmocr import model_manager
from utils.file_processor import FileProcessor


class OCRService:
    """
    Main OCR processing service.
    
    This service coordinates file processing, model operations,
    and result formatting for the RolmOCR endpoint.
    """
    
    def __init__(self):
        """Initialize the OCR service."""
        self.file_processor = FileProcessor()
        self.model_manager = model_manager
    
    def process_request(self, job_input):
        """
        Process an OCR request.
        
        Args:
            job_input (dict): Job input containing file data or URL
            
        Returns:
            dict: Processing result with extracted text and metadata
            
        Raises:
            Exception: If processing fails
        """
        try:
            # Get configuration from environment variables
            config = self._get_config()
            
            # Process file input
            file_data, file_extension = self._process_input(job_input, config)
            
            # Process file based on type
            extracted_texts = self._process_file(file_data, file_extension, config)
            
            # Format and return result
            return self._format_result(extracted_texts)
            
        except Exception as e:
            # Clean up memory on error
            self.model_manager.cleanup_memory()
            raise e
    
    def _get_config(self):
        """
        Get configuration from environment variables.
        
        Returns:
            dict: Configuration parameters
        """
        return {
            'max_tokens': int(os.getenv("MAX_TOKENS", "4096")),
            'temperature': float(os.getenv("TEMPERATURE", "0.2")),
            'max_pages': int(os.getenv("MAX_PAGES", "10")),
            'max_file_size_mb': int(os.getenv("MAX_FILE_SIZE_MB", "50"))
        }
    
    def _process_input(self, job_input, config):
        """
        Process job input to get file data and extension.
        
        Args:
            job_input (dict): Job input
            config (dict): Configuration parameters
            
        Returns:
            tuple: (file_data, file_extension)
        """
        # Handle URL input
        if "url" in job_input:
            url = job_input["url"]
            file_data = self.file_processor.download_file_from_url(
                url, config['max_file_size_mb']
            )
            file_extension = self.file_processor.detect_file_type(url=url)
            
        # Handle direct file input (backward compatibility)
        elif "file" in job_input:
            if isinstance(job_input["file"], str):
                file_data = base64.b64decode(job_input["file"])
                file_extension = job_input.get("file_extension", "pdf")
            else:
                file_data = job_input["file"]
                file_extension = job_input.get("file_extension", "pdf")
        else:
            raise ValueError("No URL or file provided in input")
        
        return file_data, file_extension
    
    def _process_file(self, file_data, file_extension, config):
        """
        Process file based on its type.
        
        Args:
            file_data (bytes): File data
            file_extension (str): File extension
            config (dict): Configuration parameters
            
        Returns:
            list: List of extracted text results
        """
        extracted_texts = []
        
        if not self.file_processor.is_supported_format(file_extension):
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        if file_extension.lower() == '.pdf':
            # Process PDF
            extracted_texts = self._process_pdf(file_data, config)
            
        elif file_extension.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
            # Process single image
            extracted_texts = self._process_image(file_data, config)
        
        return extracted_texts
    
    def _process_pdf(self, pdf_data, config):
        """
        Process PDF file.
        
        Args:
            pdf_data (bytes): PDF data
            config (dict): Configuration parameters
            
        Returns:
            list: List of extracted text results
        """
        # Convert PDF to images
        images = self.file_processor.pdf_to_images(pdf_data)
        
        # Process each page
        extracted_texts = []
        for i, image in enumerate(images):
            try:
                text = self.model_manager.process_image(
                    image, 
                    config['temperature'], 
                    config['max_tokens']
                )
                extracted_texts.append({
                    "page": i + 1,
                    "text": text
                })
                
            except Exception as e:
                extracted_texts.append({
                    "page": i + 1,
                    "text": f"Error processing page: {str(e)}"
                })
        
        return extracted_texts
    
    def _process_image(self, image_data, config):
        """
        Process single image file.
        
        Args:
            image_data (bytes): Image data
            config (dict): Configuration parameters
            
        Returns:
            list: List with single extracted text result
        """
        try:
            image = Image.open(BytesIO(image_data))
            text = self.model_manager.process_image(
                image, 
                config['temperature'], 
                config['max_tokens']
            )
            
            return [{
                "page": 1,
                "text": text
            }]
            
        except Exception as e:
            raise Exception(f"Failed to process image: {str(e)}")
    
    def _format_result(self, extracted_texts):
        """
        Format the processing result.
        
        Args:
            extracted_texts (list): List of extracted text results
            
        Returns:
            dict: Formatted result
        """
        # Combine all extracted text
        combined_text = "\n\n".join([page["text"] for page in extracted_texts])
        
        # Clean up memory after processing
        self.model_manager.cleanup_memory()
        
        return {
            "extracted_text": combined_text,
            "pages": extracted_texts,
            "total_pages": len(extracted_texts),
            "status": "success"
        }
    
    def get_service_info(self):
        """
        Get information about the OCR service.
        
        Returns:
            dict: Service information
        """
        model_info = self.model_manager.get_model_info()
        supported_formats = self.file_processor.get_supported_formats()
        
        return {
            "model": model_info,
            "supported_formats": supported_formats,
            "service": "RolmOCR Endpoint"
        }
