from pydantic import BaseModel
from typing import Dict, List

class CriteriaMatch(BaseModel):
    criterion: str
    score: float
    matches: List[str]

class AssessmentResponse(BaseModel):
    qualification_rating: str
    overall_score: float
    criteria_matches: List[CriteriaMatch]
    explanation: str
    recommendations: List[str] 