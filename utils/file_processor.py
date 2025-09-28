"""
File Processing Utilities Module

This module handles file downloads, PDF processing, and image conversion
for the RolmOCR endpoint.
"""

import os
import requests
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF
from urllib.parse import urlparse


class FileProcessor:
    """
    Handles file processing operations including downloads and conversions.
    
    Features:
    - URL file downloads with size limits
    - PDF to image conversion
    - File type detection
    - Error handling and validation
    """
    
    def __init__(self, max_file_size_mb=50, max_pages=10):
        """
        Initialize the file processor.
        
        Args:
            max_file_size_mb (int): Maximum file size in MB
            max_pages (int): Maximum pages to process from PDF
        """
        self.max_file_size_mb = max_file_size_mb
        self.max_pages = max_pages
    
    def download_file_from_url(self, url, timeout=30):
        """
        Download file from URL with size and timeout limits.
        
        Args:
            url (str): URL to download from
            timeout (int): Request timeout in seconds
            
        Returns:
            bytes: File data
            
        Raises:
            Exception: If download fails or file is too large
        """
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError("Invalid URL provided")
            
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            
            # Check content length
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_file_size_mb * 1024 * 1024:
                raise ValueError(f"File too large. Maximum size: {self.max_file_size_mb}MB")
            
            # Download file in chunks
            file_data = BytesIO()
            downloaded_size = 0
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    downloaded_size += len(chunk)
                    if downloaded_size > self.max_file_size_mb * 1024 * 1024:
                        raise ValueError(f"File too large. Maximum size: {self.max_file_size_mb}MB")
                    file_data.write(chunk)
            
            file_data.seek(0)
            return file_data.getvalue()
            
        except Exception as e:
            raise Exception(f"Failed to download file from URL: {str(e)}")
    
    def pdf_to_images(self, pdf_data):
        """
        Convert PDF data to list of PIL Images.
        
        Args:
            pdf_data (bytes): PDF file data
            
        Returns:
            list: List of PIL Images
            
        Raises:
            Exception: If PDF conversion fails
        """
        try:
            # Open PDF from bytes
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            images = []
            
            # Process up to max_pages
            pages_to_process = min(len(pdf_document), self.max_pages)
            
            for page_num in range(pages_to_process):
                page = pdf_document[page_num]
                
                # Render page to image with 2x zoom for better quality
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                
                # Convert to PIL Image
                image = Image.open(BytesIO(img_data))
                images.append(image)
            
            pdf_document.close()
            return images
            
        except Exception as e:
            raise Exception(f"Failed to convert PDF to images: {str(e)}")
    
    def detect_file_type(self, url=None, file_extension=None):
        """
        Detect file type from URL or extension.
        
        Args:
            url (str, optional): URL to detect type from
            file_extension (str, optional): File extension
            
        Returns:
            str: Detected file extension (e.g., '.pdf', '.png')
        """
        if file_extension:
            return file_extension.lower()
        
        if url:
            parsed_url = urlparse(url)
            extension = os.path.splitext(parsed_url.path)[1].lower()
            if extension:
                return extension
            
            # Try to detect from content type
            try:
                response = requests.head(url, timeout=10)
                content_type = response.headers.get('content-type', '')
                if 'pdf' in content_type:
                    return '.pdf'
                elif 'image' in content_type:
                    return '.png'
            except:
                pass
        
        # Default to PDF
        return '.pdf'
    
    def is_supported_format(self, file_extension):
        """
        Check if file format is supported.
        
        Args:
            file_extension (str): File extension to check
            
        Returns:
            bool: True if supported, False otherwise
        """
        supported_formats = {
            '.pdf': 'PDF documents',
            '.png': 'PNG images',
            '.jpg': 'JPEG images',
            '.jpeg': 'JPEG images',
            '.gif': 'GIF images',
            '.bmp': 'BMP images',
            '.tiff': 'TIFF images'
        }
        
        return file_extension.lower() in supported_formats
    
    def get_supported_formats(self):
        """
        Get list of supported file formats.
        
        Returns:
            dict: Dictionary of supported formats
        """
        return {
            'documents': ['.pdf'],
            'images': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
        }
