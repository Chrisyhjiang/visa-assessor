from pydantic import BaseModel
from enum import Enum
from typing import List, Dict, Optional

class QualificationRating(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class CriterionMatch(BaseModel):
    criterion: str
    evidence: List[str]
    confidence: float

class AssessmentResponse(BaseModel):
    criteria_matches: Dict[str, CriterionMatch]
    overall_rating: QualificationRating
    explanation: str 