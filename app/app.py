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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class CriterionEvidence(BaseModel):
    score: float
    evidence: List[str]

class AssessmentResponse(BaseModel):
    criteria_matches: Dict[str, CriterionEvidence]
    qualification_rating: str
    overall_score: float
    explanation: Optional[str] = None
    recommendations: Optional[List[str]] = None
    agent_explanation: Optional[str] = None
    agent_recommendations: Optional[List[str]] = None

@app.get("/")
async def root():
    return {"message": "Welcome to O-1A Visa Assessor API. Use /docs for API documentation."}

@app.post("/assess-visa", response_model=AssessmentResponse)
async def assess_visa(cv_file: UploadFile = File(...)):
    """
    Upload a CV file and get an ML-based assessment for O-1A visa qualification.
    
    Returns:
    - criteria_matches: Dictionary of criteria with confidence scores and matching evidence
    - qualification_rating: Overall qualification rating (low, medium, high)
    - overall_score: Aggregate score across all criteria
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