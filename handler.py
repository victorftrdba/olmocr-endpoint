"""
RolmOCR RunPod Serverless Handler

This is the main entry point for the RunPod serverless endpoint.
It initializes the OCR service and handles incoming requests.
"""

import runpod
from models.rolmocr import model_manager
from services.ocr_service import OCRService


# Initialize OCR service
try:
    model_manager.load_model()
except Exception as e:
    # Don't raise here - let the service handle it gracefully
    pass

# Create OCR service instance
ocr_service = OCRService()


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
        # Get job input
        job_input = job.get("input", {})
        
        if not job_input:
            return {
                "error": "No input provided",
                "status": "error"
            }
        
        # Check if model is loaded
        if not model_manager.model:
            try:
                model_manager.load_model()
            except Exception as model_error:
                return {
                    "error": f"Model not available: {str(model_error)}",
                    "status": "error"
                }
        
        # Process the request using OCR service
        result = ocr_service.process_request(job_input)
        return result
        
    except Exception as e:
        # Clean up memory on error
        model_manager.cleanup_memory()
        return {
            "error": str(e),
            "status": "error"
        }

# Start the RunPod serverless handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
