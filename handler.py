"""
RolmOCR RunPod Serverless Handler

This is the main entry point for the RunPod serverless endpoint.
It initializes the OCR service and handles incoming requests.
"""

import runpod
from models.rolmocr import model_manager
from services.ocr_service import OCRService

# Global variables for lazy initialization
ocr_service = None
model_loaded = False


def initialize_service():
    """
    Initialize the OCR service and model with proper error handling.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    global ocr_service, model_loaded
    
    if model_loaded and ocr_service is not None:
        return True
    
    try:
        # Load model with timeout protection
        model_manager.load_model()
        
        # Create OCR service instance
        ocr_service = OCRService()
        
        model_loaded = True
        return True
        
    except Exception as e:
        model_loaded = False
        ocr_service = None
        return False


def handler(job):
    """
    RunPod serverless handler for OCR processing.
    
    This function is called for each incoming request and delegates
    the processing to the OCR service.
    
    Args:
        job (dict): RunPod job containing input data
        
    Returns:
        dict: Processing result with extracted text and metadata
    """
    try:
        # Initialize service if not already done
        if not initialize_service():
            return {
                "error": "Failed to initialize OCR service",
                "status": "error"
            }
        
        # Get job input
        job_input = job.get("input", {})
        
        if not job_input:
            return {
                "error": "No input provided",
                "status": "error"
            }
        
        # Process the request using OCR service
        result = ocr_service.process_request(job_input)
        return result
        
    except Exception as e:
        # Clean up memory on error
        try:
            model_manager.cleanup_memory()
        except:
            pass
        return {
            "error": str(e),
            "status": "error"
        }

# Start the RunPod serverless handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
