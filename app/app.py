"""
O-1A Visa Assessor API

This is the main FastAPI application that provides endpoints for assessing O-1A visa qualifications
based on CV analysis. The application uses OpenAI's GPT models to analyze uploaded CVs and provide
detailed assessments of how well the applicant meets O-1A visa criteria.

Key Features:
- CV upload and processing (PDF, DOCX, TXT formats)
- Detailed analysis of O-1A visa criteria matches
- Qualification rating and scoring
- Evidence-based recommendations
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
from app.services.visa_assessor import VisaAssessor

app = FastAPI(
    title="O-1A Visa Assessor API",
    description="ML-based API to assess O-1A visa qualifications based on CV analysis",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class CriterionEvidence(BaseModel):
    """
    Pydantic model for storing evidence for a specific O-1A criterion.
    
    Attributes:
        score (float): Confidence score for the criterion match (0-1)
        evidence (List[str]): List of text excerpts from CV supporting the criterion
    """
    score: float
    evidence: List[str]

class AssessmentResponse(BaseModel):
    """
    Pydantic model for the complete visa assessment response.
    
    Attributes:
        criteria_matches (Dict[str, CriterionEvidence]): Matches for each O-1A criterion
        qualification_rating (str): Overall rating (high/medium/low)
        overall_score (float): Aggregate score across all criteria
        explanation (Optional[str]): Detailed explanation of the assessment
        recommendations (Optional[List[str]]): Suggested improvements
        agent_explanation (Optional[str]): USCIS officer-style explanation
        agent_recommendations (Optional[List[str]]): Officer-style recommendations
    """
    criteria_matches: Dict[str, CriterionEvidence]
    qualification_rating: str
    overall_score: float
    explanation: Optional[str] = None
    recommendations: Optional[List[str]] = None
    agent_explanation: Optional[str] = None
    agent_recommendations: Optional[List[str]] = None

@app.get("/")
async def root():
    """
    Root endpoint that provides basic API information.
    
    Returns:
        dict: Welcome message and link to API documentation
    """
    return {"message": "Welcome to O-1A Visa Assessor API. Use /docs for API documentation."}

@app.post("/assess-visa", response_model=AssessmentResponse)
async def assess_visa(cv_file: UploadFile = File(...)):
    """
    Endpoint for uploading and assessing a CV for O-1A visa qualification.
    
    Args:
        cv_file (UploadFile): The CV file to be analyzed (PDF, DOCX, or TXT format)
    
    Returns:
        AssessmentResponse: Detailed assessment of O-1A visa qualification
        
    Raises:
        HTTPException: If file format is unsupported or processing fails
    """
    
    # Check file type (accept PDF, DOCX, TXT)
    allowed_extensions = ["pdf", "docx", "txt"]
    file_extension = cv_file.filename.split(".")[-1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Please upload a {', '.join(allowed_extensions)} file."
        )
    
    # Read the file content
    content = await cv_file.read()
    
    try:
        # Process the CV using ML-based assessment
        assessor = VisaAssessor()
        assessment = assessor.assess_cv(content, file_extension)
        
        return assessment
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CV: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 