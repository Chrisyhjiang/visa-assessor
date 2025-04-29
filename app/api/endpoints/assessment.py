from fastapi import APIRouter, HTTPException
from typing import Dict
from app.services.assessment_service import AssessmentService
from app.schemas.assessment import AssessmentResponse

router = APIRouter()
assessment_service = AssessmentService()

@router.post("/assess", response_model=AssessmentResponse)
async def assess_application(application_data: Dict):
    try:
        assessment = assessment_service.assess_application(application_data)
        return AssessmentResponse(
            qualification_rating=assessment['qualification_rating'],
            overall_score=assessment['overall_score'],
            criteria_matches=assessment['criteria_matches'],
            explanation=assessment['explanation'],
            recommendations=assessment['recommendations']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 