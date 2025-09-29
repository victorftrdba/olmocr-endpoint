"""
Configuration Module for RolmOCR Endpoint

This module contains configuration settings and environment variable
definitions for the RolmOCR endpoint.
"""

import os


class Config:
    """
    Configuration class for RolmOCR endpoint.
    
    This class manages all configuration settings including
    model parameters, file processing limits, and API settings.
    """
    
    # Model Configuration
    MODEL_NAME = "reducto/RolmOCR"
    PROCESSOR_NAME = "reducto/RolmOCR"
    
    # Processing Parameters
    DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
    DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
    DEFAULT_MAX_PAGES = int(os.getenv("MAX_PAGES", "10"))
    DEFAULT_MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    DEFAULT_PROCESSING_TIMEOUT = int(os.getenv("PROCESSING_TIMEOUT", "120"))  # 2 minutes
    
    # File Processing
    DOWNLOAD_TIMEOUT = 30
    CHUNK_SIZE = 8192
    PDF_ZOOM_FACTOR = 2.0
    
    # Supported File Formats
    SUPPORTED_DOCUMENT_FORMATS = ['.pdf']
    SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
    
    @classmethod
    def get_supported_formats(cls):
        """
        Get all supported file formats.
        
        Returns:
            list: List of all supported file extensions
        """
        return cls.SUPPORTED_DOCUMENT_FORMATS + cls.SUPPORTED_IMAGE_FORMATS
    
    @classmethod
    def get_model_config(cls):
        """
        Get model configuration.
        
        Returns:
            dict: Model configuration parameters
        """
        return {
            'model_name': cls.MODEL_NAME,
            'processor_name': cls.PROCESSOR_NAME
        }
    
    @classmethod
    def get_processing_config(cls):
        """
        Get processing configuration.
        
        Returns:
            dict: Processing configuration parameters
        """
        return {
            'temperature': cls.DEFAULT_TEMPERATURE,
            'max_tokens': cls.DEFAULT_MAX_TOKENS,
            'max_pages': cls.DEFAULT_MAX_PAGES,
            'max_file_size_mb': cls.DEFAULT_MAX_FILE_SIZE_MB,
            'processing_timeout': cls.DEFAULT_PROCESSING_TIMEOUT,
            'download_timeout': cls.DOWNLOAD_TIMEOUT,
            'chunk_size': cls.CHUNK_SIZE,
            'pdf_zoom_factor': cls.PDF_ZOOM_FACTOR
        }
    
    @classmethod
    def validate_config(cls):
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        if cls.DEFAULT_TEMPERATURE < 0.0 or cls.DEFAULT_TEMPERATURE > 1.0:
            raise ValueError("DEFAULT_TEMPERATURE must be between 0.0 and 1.0")
        
        if cls.DEFAULT_MAX_TOKENS <= 0:
            raise ValueError("DEFAULT_MAX_TOKENS must be positive")
        
        if cls.DEFAULT_MAX_PAGES <= 0:
            raise ValueError("DEFAULT_MAX_PAGES must be positive")
        
        if cls.DEFAULT_MAX_FILE_SIZE_MB <= 0:
            raise ValueError("DEFAULT_MAX_FILE_SIZE_MB must be positive")
        
        if cls.DEFAULT_PROCESSING_TIMEOUT <= 0:
            raise ValueError("DEFAULT_PROCESSING_TIMEOUT must be positive")
