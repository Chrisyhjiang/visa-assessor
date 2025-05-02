"""
Assessment Schema

This module defines the Pydantic models used for structuring and validating
the assessment data in the O-1A Visa Assessor application. These models ensure
that the data passed between different components of the application maintains
a consistent structure and type safety.
"""

from pydantic import BaseModel
from typing import Dict, List

class CriteriaMatch(BaseModel):
    """
    Model representing a match for a specific O-1A visa criterion.
    
    This model captures both the quantitative score and qualitative evidence
    for how well an applicant meets a particular criterion.
    
    Attributes:
        criterion (str): Name of the O-1A criterion being evaluated
        score (float): Confidence score for the criterion match (0-1)
        matches (List[str]): Text excerpts from CV supporting this criterion
    """
    criterion: str
    score: float
    matches: List[str]

class AssessmentResponse(BaseModel):
    """
    Model representing the complete assessment response.
    
    This model structures the overall assessment of an O-1A visa application,
    including both the detailed criterion matches and the overall evaluation.
    
    Attributes:
        qualification_rating (str): Overall rating (high/medium/low)
        overall_score (float): Aggregate score across all criteria (0-1)
        criteria_matches (List[CriteriaMatch]): Detailed matches for each criterion
        explanation (str): Detailed explanation of the assessment
        recommendations (List[str]): Suggested improvements for the application
    """
    qualification_rating: str
    overall_score: float
    criteria_matches: List[CriteriaMatch]
    explanation: str
    recommendations: List[str] 