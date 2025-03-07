# Set environment variables before any imports
import os
import platform

# Configure GPU acceleration based on platform
if platform.system() == "Darwin":  # macOS
    # Force CPU fallback for MPS issues
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    # Set watermark ratio to 0.0 to avoid the invalid ratio error
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Enable RAG by default
os.environ["USE_RAG"] = "true"

# Import necessary libraries
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import torch  # Import torch after environment variables are set
from typing import Dict, Any, Optional, List
import tempfile
import logging
import asyncio
from starlette.responses import StreamingResponse
import uuid
import json
import argparse

# Configure GPU acceleration based on platform (print messages only)
if platform.system() == "Darwin":  # macOS
    print("Detected macOS system, configuring for MPS acceleration")
    print("MPS environment variables set to avoid watermark ratio issues")
    
    # Check if MPS is available
    try:
        # Correct way to check for MPS availability
        if torch.backends.mps.is_available():
            print("MPS (Metal Performance Shaders) is available")
        else:
            print("MPS is not available on this system, using CPU instead")
    except Exception as e:
        print(f"Error checking MPS availability: {str(e)}")
        print("Using CPU for inference")
elif platform.system() == "Linux" or platform.system() == "Windows":
    print("Detected Linux/Windows system, configuring for CUDA acceleration if available")
    # CUDA settings are handled automatically by PyTorch

# Import our processing modules
from processors.document_parser import extract_text_from_document
from processors.prompt_builder import build_o1a_assessment_prompt, build_o1a_assessment_prompt_with_rag
from processors.response_formatter import format_llm_response
from model_setup import initialize_model, generate_text, safe_tensor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="O-1A Visa Qualification Assessment API",
    description="API for assessing O-1A visa qualification based on CV analysis",
    version="1.0.0"
)

# Global variables for model and RAG
model = None
model_type = None
model_name = None
retriever = None
use_rag = False
# Store processing results
processing_results = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the model and other resources on startup"""
    global model, model_type, retriever, use_rag
    
    # Check if we should use RAG
    use_rag = os.environ.get("USE_RAG", "true").lower() == "true"
    
    # Initialize the model
    try:
        logger.info("Initializing model...")
        model, model_type = initialize_model()
        # Get the model name from the pipeline
        if hasattr(model, "model") and hasattr(model.model, "config"):
            model_name = model.model.config.name_or_path
        else:
            model_name = "Unknown model"
        logger.info(f"Model initialized successfully: {model_name} (type: {model_type})")
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        
        # Create a dummy model function that returns a fixed response
        logger.warning("Creating a dummy model as fallback")
        
        def dummy_model(prompt, **kwargs):
            return [{"generated_text": prompt + "\n\nI'm sorry, but the language model is currently unavailable. Please try again later."}]
        
        model = dummy_model
        model_type = "dummy"
        logger.info("Dummy model created as fallback")
    
    # Initialize RAG if enabled
    if use_rag:
        try:
            # Use the simple retriever which doesn't have dependency issues
            from rag.simple_retriever import SimpleRetriever
            logger.info("Initializing simple RAG retriever...")
            retriever = SimpleRetriever(chunks_file="data/rag/legal_chunks.json")
            logger.info("Simple RAG retriever initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG: {str(e)}")
            use_rag = False
            logger.warning("RAG disabled due to initialization error")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    global use_rag, model_type, model_name
    
    return {
        "message": "O-1A Visa Qualification Assessment API",
        "version": "1.0.0",
        "status": "active",
        "config": {
            "rag_enabled": use_rag,
            "model_type": model_type,
            "model_name": model_name
        },
        "endpoints": {
            "/assess": "POST - Assess O-1A qualification from CV"
        }
    }

@app.post("/assess")
async def assess_qualification(
    cv_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Assess O-1A visa qualification based on uploaded CV
    """
    global model, model_type, retriever, use_rag, processing_results
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # Create temp file to store the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(cv_file.filename)[1]) as temp_file:
        content = await cv_file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    # Generate a unique ID for this request
    request_id = str(uuid.uuid4())
    
    # Start processing in background
    background_tasks.add_task(
        process_document_in_background, 
        temp_file_path, 
        request_id, 
        model, 
        model_type, 
        retriever, 
        use_rag
    )
    
    # Return the request ID immediately
    return {"request_id": request_id, "status": "processing"}

@app.get("/assess/status/{request_id}")
async def get_assessment_status(request_id: str):
    """Check the status of an assessment request"""
    global processing_results
    
    if request_id not in processing_results:
        return {"status": "not_found"}
    
    result = processing_results[request_id]
    if result.get("status") == "completed":
        # If completed, return the result and remove from memory
        response = result.copy()
        # Keep results in memory for a while, could implement cleanup with TTL
        return response
    else:
        # Still processing
        return {"status": "processing"}

async def process_document_in_background(
    temp_file_path: str, 
    request_id: str,
    model,
    model_type,
    retriever,
    use_rag
):
    """Process document in background"""
    global processing_results
    
    # Initialize result with processing status
    processing_results[request_id] = {"status": "processing"}
    
    try:
        # Extract text from document
        cv_text = extract_text_from_document(temp_file_path)
        logger.info(f"Extracted {len(cv_text)} characters from CV")
        
        # Build prompt based on RAG availability
        if use_rag and retriever is not None:
            # Use the simple retriever's get_relevant_legal_context function
            from rag.simple_retriever import get_relevant_legal_context
            legal_context = get_relevant_legal_context(retriever, cv_text)
            prompt = build_o1a_assessment_prompt_with_rag(cv_text, legal_context)
            logger.info("Using RAG-enhanced prompt")
        else:
            prompt = build_o1a_assessment_prompt(cv_text)
            logger.info("Using standard prompt")
        
        # Run inference with the model
        logger.info(f"Running inference with model (type: {model_type})")
        
        # Measure inference time
        import time
        start_time = time.time()
        
        llm_response = generate_text(model, prompt, model_type)
        
        end_time = time.time()
        inference_time = end_time - start_time
        logger.info(f"Inference completed in {inference_time:.2f} seconds")
        
        # Clean and format the response
        formatted_response = format_llm_response(llm_response)
        
        # Store the result
        processing_results[request_id] = {
            "status": "completed",
            "result": formatted_response,
            "metrics": {
                "inference_time_seconds": round(inference_time, 2)
            }
        }
        
        # Clean up the temp file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        # Store the error
        processing_results[request_id] = {
            "status": "error",
            "error": str(e)
        }
        # Clean up the temp file in case of error
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def check_ports(port):
    """Check if a port is in use and find an available one if needed"""
    import socket
    
    # Check if the port is in use
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("0.0.0.0", port))
        s.close()
        return port  # Port is available
    except socket.error:
        logger.warning(f"Port {port} is already in use, trying to find an available port...")
        
        # Try to find an available port
        for test_port in range(port + 1, port + 100):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("0.0.0.0", test_port))
                s.close()
                logger.info(f"Found available port: {test_port}")
                return test_port
            except socket.error:
                continue
        
        # If we get here, we couldn't find an available port
        logger.error(f"Could not find an available port in range {port}-{port+100}")
        return port  # Return the original port and let uvicorn handle the error

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="O-1A Visa Qualification Assessment API")
    parser.add_argument("--port", type=int, default=9000, help="Port to run the API server on")
    args = parser.parse_args()
    
    # Get port from arguments or environment
    requested_port = int(os.environ.get("PORT", args.port))
    
    # Check if the port is available and find an alternative if needed
    port = check_ports(requested_port)
    
    # Print startup message
    print(f"=== O-1A Visa Qualification Assessment API ===")
    print(f"Starting server on http://localhost:{port}")
    print(f"Press Ctrl+C to stop the server.")
    
    # Run the server
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True) 