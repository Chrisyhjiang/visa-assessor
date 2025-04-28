from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import logging
from typing import Dict, Any

from app.models.assessment import AssessmentResponse, CriterionMatch, QualificationRating
from app.services.cv_parser import parse_cv
from app.services.assessment import O1AAssessmentService, assess_o1a_qualification

# Set up logging
logger = logging.getLogger(__name__)

app = FastAPI(
    title="O-1A Visa Qualification Assessment API",
    description="API for assessing O-1A visa qualification based on CV analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service instances
_assessment_service = None
_o1a_service_instance = None

def get_o1a_service():
    """Get or initialize the O1A service instance."""
    global _o1a_service_instance
    if _o1a_service_instance is None:
        try:
            logger.info("Initializing O1A Assessment Service...")
            _o1a_service_instance = O1AAssessmentService()
            logger.info("O1A Assessment Service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize O1A service: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize O1A service: {str(e)}")
    return _o1a_service_instance

# Initialize assessment service at startup
@app.on_event("startup")
async def startup_event():
    """Initialize services at startup."""
    try:
        # Pre-initialize the O1A service instance
        _ = get_o1a_service()
    except Exception as e:
        logger.error(f"Failed to initialize services at startup: {str(e)}", exc_info=True)
        # We don't raise an exception here to allow the server to start

@app.post("/assess-cv", response_model=AssessmentResponse)
async def assess_cv(file: UploadFile = File(...)):
    """
    Assess a CV for O-1A visa qualification.
    
    - **file**: CV file (PDF, DOCX, or TXT format)
    
    Returns an assessment of the CV against O-1A visa criteria.
    """
    logger.info(f"Received file: {file.filename}")
    temp_file_path = None
    
    try:
        # Validate file format
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in ['.pdf', '.docx', '.txt']:
            logger.warning(f"Unsupported file format: {file_extension}")
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file format. Please upload a PDF, DOCX, or TXT file."
            )
        
        # Save the uploaded file temporarily
        temp_file_path = await save_uploaded_file(file, file_extension)
        
        # Parse the CV
        logger.info("Parsing CV")
        cv_text = parse_cv(temp_file_path)
        logger.info(f"Successfully parsed CV, extracted {len(cv_text)} characters")
        
        # Assess the CV against O-1A criteria
        logger.info("Assessing CV against O-1A criteria")
        assessment_result = assess_o1a_qualification(cv_text)
        logger.info("Successfully assessed CV")
        
        # Convert the assessment result to the response model format
        response = create_assessment_response(assessment_result)
        logger.info("Returning assessment response")
        return response
        
    except Exception as e:
        logger.error(f"Error processing CV: {str(e)}", exc_info=True)
        error_message = get_user_friendly_error_message(e)
        raise HTTPException(status_code=500, detail=error_message)
    
    finally:
        # Clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                logger.info("Cleaning up temporary file")
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up temporary file: {str(cleanup_error)}")

async def save_uploaded_file(file: UploadFile, file_extension: str) -> str:
    """Save the uploaded file to a temporary location."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        content = await file.read()
        logger.info(f"Read {len(content)} bytes from uploaded file")
        temp_file.write(content)
        return temp_file.name

def create_assessment_response(assessment_result: Dict[str, Any]) -> AssessmentResponse:
    """Create an AssessmentResponse from the assessment result."""
    criteria_matches = {}
    for criterion, result in assessment_result["criteria_matches"].items():
        criteria_matches[criterion] = CriterionMatch(
            criterion=result["criterion"],
            evidence=result["evidence"],
            confidence=result["confidence"]
        )
    
    return AssessmentResponse(
        criteria_matches=criteria_matches,
        overall_rating=assessment_result["overall_rating"],
        explanation=assessment_result["explanation"]
    )

def get_user_friendly_error_message(e: Exception) -> str:
    """Create a user-friendly error message."""
    error_str = str(e)
    
    # Check for specific error types
    if "sentence_transformers" in error_str:
        return "Server configuration error: Missing required dependencies. Please contact the administrator."
    elif "File not found" in error_str or "Could not open" in error_str:
        return "Error processing file: The file could not be read or is corrupted."
    elif "model" in error_str.lower() and ("load" in error_str.lower() or "download" in error_str.lower()):
        return "Server configuration error: Unable to load AI models. Please contact the administrator."
    else:
        return f"Error processing CV: {error_str}"

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 