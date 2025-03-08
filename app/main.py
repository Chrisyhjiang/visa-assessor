from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import logging
import traceback
from typing import Dict, Any

from app.models.assessment import AssessmentResponse, CriterionMatch, QualificationRating
from app.services.cv_parser import parse_cv

# Set up logging
logging.basicConfig(level=logging.INFO)
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

# Global variable for assessment service
assessment_service = None
o1a_service_instance = None

# Initialize assessment service at startup
@app.on_event("startup")
async def startup_event():
    global assessment_service, o1a_service_instance
    try:
        logger.info("Initializing assessment service at startup...")
        from app.services.assessment import assess_o1a_qualification, O1AAssessmentService
        assessment_service = assess_o1a_qualification
        
        # Pre-initialize the O1A service instance
        logger.info("Pre-initializing O1A Assessment Service...")
        o1a_service_instance = O1AAssessmentService()
        logger.info("O1A Assessment Service initialized successfully at startup")
    except Exception as e:
        logger.error(f"Failed to initialize assessment service at startup: {str(e)}", exc_info=True)
        # We don't raise an exception here to allow the server to start
        # The error will be handled when endpoints are called

def get_assessment_service():
    global assessment_service
    if assessment_service is None:
        try:
            from app.services.assessment import assess_o1a_qualification
            assessment_service = assess_o1a_qualification
        except ImportError as e:
            logger.error(f"Failed to import assessment service: {str(e)}", exc_info=True)
            raise ImportError(f"Failed to import assessment service: {str(e)}")
    return assessment_service

def get_o1a_service():
    global o1a_service_instance
    if o1a_service_instance is None:
        try:
            from app.services.assessment import O1AAssessmentService
            o1a_service_instance = O1AAssessmentService()
        except Exception as e:
            logger.error(f"Failed to initialize O1A service: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize O1A service: {str(e)}")
    return o1a_service_instance

@app.post("/assess-cv", response_model=AssessmentResponse)
async def assess_cv(file: UploadFile = File(...)):
    """
    Assess a CV for O-1A visa qualification.
    
    - **file**: CV file (PDF or DOCX format)
    
    Returns an assessment of the CV against O-1A visa criteria.
    """
    logger.info(f"Received file: {file.filename}")
    temp_file_path = None
    
    try:
        # Check file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in ['.pdf', '.docx', '.txt']:
            logger.warning(f"Unsupported file format: {file_extension}")
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file format. Please upload a PDF, DOCX, or TXT file."
            )
        
        # Save the uploaded file temporarily
        logger.info("Saving uploaded file temporarily")
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            logger.info(f"Read {len(content)} bytes from uploaded file")
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Parse the CV
            logger.info("Parsing CV")
            cv_text = parse_cv(temp_file_path)
            logger.info(f"Successfully parsed CV, extracted {len(cv_text)} characters")
            
            # Use the pre-initialized assessment service
            try:
                assess_func = get_assessment_service()
            except ImportError as e:
                logger.error(f"Dependency error: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail="Server configuration error: Missing required dependencies. Please check server logs."
                )
            
            # Assess the CV against O-1A criteria
            logger.info("Assessing CV against O-1A criteria")
            assessment_result = assess_func(cv_text)
            logger.info("Successfully assessed CV")
            
            # Convert the assessment result to the response model format
            criteria_matches = {}
            for criterion, result in assessment_result["criteria_matches"].items():
                criteria_matches[criterion] = CriterionMatch(
                    criterion=result["criterion"],
                    evidence=result["evidence"],
                    confidence=result["confidence"]
                )
            
            response = AssessmentResponse(
                criteria_matches=criteria_matches,
                overall_rating=assessment_result["overall_rating"],
                explanation=assessment_result["explanation"]
            )
            
            logger.info("Returning assessment response")
            return response
            
        finally:
            # Clean up the temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                logger.info("Cleaning up temporary file")
                os.unlink(temp_file_path)
    
    except Exception as e:
        logger.error(f"Error processing CV: {str(e)}", exc_info=True)
        # Clean up the temporary file if it exists
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up temporary file: {str(cleanup_error)}")
        
        # Provide a more user-friendly error message for dependency issues
        if "sentence_transformers" in str(e):
            error_message = "Server configuration error: The sentence_transformers package is required but not properly installed. Please contact the administrator."
        else:
            error_message = f"Error processing CV: {str(e)}"
        
        raise HTTPException(status_code=500, detail=error_message)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 