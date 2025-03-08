from typing import Dict, List, Any, Optional
from enum import Enum
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from app.services.rag_service import RAGService
from app.services.cv_parser import parse_cv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualificationRating(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class CriterionMatch:
    def __init__(self, criterion: str, evidence: List[str], confidence: float):
        self.criterion = criterion
        self.evidence = evidence
        self.confidence = confidence

class O1AAssessmentService:
    def __init__(self):
        """Initialize the O-1A assessment service with Qwen model."""
        try:
            # Initialize the RAG service
            logger.info("Initializing RAG service...")
            self.rag_service = RAGService()
            
            # Initialize Qwen2.5 model (smallest version for speed)
            logger.info("Initializing Qwen2.5 model...")
            self.model_name = "Qwen/Qwen2.5-0.5B"
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
                logger.info("Qwen2.5 model initialized successfully")
            except Exception as e:
                # Fallback to a more commonly available model if Qwen fails
                logger.warning(f"Failed to load Qwen2.5 model: {str(e)}. Falling back to gpt2.")
                self.model_name = "gpt2"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                logger.info("Fallback model initialized successfully")
            
            # Set the criteria
            self.criteria = [
                "Awards", 
                "Membership", 
                "Press", 
                "Judging", 
                "Original_contribution",
                "Scholarly_articles", 
                "Critical_employment", 
                "High_remuneration"
            ]
            logger.info("O1A Assessment Service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing O1A Assessment Service: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize O1A Assessment Service: {str(e)}")
    
    def _generate_response(self, prompt: str, max_length: int = 512) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: Input prompt for the model
            max_length: Maximum length of the generated response
            
        Returns:
            Generated response text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,  # Enable sampling for more diverse outputs
                temperature=0.8,  # Slightly higher temperature for more diversity
                top_p=0.92,
                repetition_penalty=1.2,  # Penalize repetition
                no_repeat_ngram_size=3,  # Prevent repeating 3-grams
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,  # Set pad token explicitly
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response
    
    def _analyze_criterion(self, cv_text: str, criterion: str, criterion_info: str) -> Dict[str, Any]:
        """
        Analyze a CV for a specific O-1A criterion.
        
        Args:
            cv_text: Text content of the CV
            criterion: The criterion to analyze
            criterion_info: Information about the criterion
            
        Returns:
            Analysis results for the criterion
        """
        prompt = f"""
        You are an expert immigration consultant specializing in O-1A visas. 
        
        CRITERION INFORMATION:
        {criterion_info}
        
        CV CONTENT:
        {cv_text}
        
        TASK:
        1. Identify any evidence in the CV that satisfies the {criterion} criterion for an O-1A visa.
        2. List each piece of evidence you find.
        3. Provide a confidence score (0-100) for how strongly this evidence satisfies the criterion.
        4. If no evidence is found, state "No evidence found."
        
        FORMAT YOUR RESPONSE AS FOLLOWS:
        Evidence:
        - [First piece of evidence]
        - [Second piece of evidence]
        - ...
        
        Confidence: [0-100]
        """
        
        response = self._generate_response(prompt)
        
        # Parse the response to extract evidence and confidence
        evidence = []
        confidence = 0
        
        if "No evidence found" not in response:
            # Extract evidence items
            if "Evidence:" in response:
                evidence_section = response.split("Evidence:")[1].split("Confidence:")[0].strip()
                evidence = [item.strip().lstrip("- ") for item in evidence_section.split("\n") if item.strip()]
            
            # Extract confidence score
            if "Confidence:" in response:
                confidence_text = response.split("Confidence:")[1].strip()
                try:
                    confidence = float(confidence_text) / 100.0  # Normalize to 0-1
                except ValueError:
                    confidence = 0.0
        
        return {
            "criterion": criterion,
            "evidence": evidence,
            "confidence": confidence
        }
    
    def _determine_overall_rating(self, criteria_results: Dict[str, Dict[str, Any]]) -> QualificationRating:
        """
        Determine the overall O-1A qualification rating based on criteria results.
        
        Args:
            criteria_results: Results for each criterion
            
        Returns:
            Overall qualification rating
        """
        # Count criteria with evidence
        criteria_with_evidence = sum(1 for result in criteria_results.values() if result["evidence"])
        
        # Calculate average confidence across criteria with evidence
        confidences = [result["confidence"] for result in criteria_results.values() if result["evidence"]]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # O-1A requires meeting at least 3 criteria
        if criteria_with_evidence >= 3 and avg_confidence > 0.7:
            return QualificationRating.HIGH
        elif criteria_with_evidence >= 3 and avg_confidence > 0.4:
            return QualificationRating.MEDIUM
        elif criteria_with_evidence >= 2:
            return QualificationRating.LOW
        else:
            return QualificationRating.LOW
    
    def _generate_explanation(self, criteria_results: Dict[str, Dict[str, Any]], overall_rating: QualificationRating) -> str:
        """
        Generate an explanation for the assessment results.
        
        Args:
            criteria_results: Results for each criterion
            overall_rating: Overall qualification rating
            
        Returns:
            Explanation text
        """
        criteria_met = [criterion for criterion, result in criteria_results.items() if result["evidence"]]
        criteria_not_met = [criterion for criterion, result in criteria_results.items() if not result["evidence"]]
        
        prompt = f"""
        You are an expert immigration consultant specializing in O-1A visas.
        
        ASSESSMENT RESULTS:
        - Overall Rating: {overall_rating.value}
        - Criteria Met: {', '.join(criteria_met) if criteria_met else 'None'}
        - Criteria Not Met: {', '.join(criteria_not_met) if criteria_not_met else 'None'}
        
        TASK:
        Write a concise explanation (3-5 sentences) of this O-1A visa qualification assessment. 
        Explain why the applicant received this rating and what it means for their chances.
        For an O-1A visa, an applicant must satisfy at least 3 of the 8 criteria.
        """
        
        explanation = self._generate_response(prompt)
        return explanation
    
    def assess_cv(self, cv_text: str) -> Dict[str, Any]:
        """
        Assess a CV for O-1A visa qualification.
        
        Args:
            cv_text: Text content of the CV
            
        Returns:
            Assessment results
        """
        criteria_results = {}
        
        # Process each criterion
        for criterion in self.criteria:
            # Get criterion information from the knowledge base
            kb_results = self.rag_service.query_knowledge_base(f"What qualifies as {criterion} for O-1A visa?", top_k=1)
            criterion_info = kb_results[0]["content"] if kb_results else ""
            
            # Analyze the CV for this criterion
            result = self._analyze_criterion(cv_text, criterion, criterion_info)
            criteria_results[criterion] = result
        
        # Determine overall rating
        overall_rating = self._determine_overall_rating(criteria_results)
        
        # Generate explanation
        explanation = self._generate_explanation(criteria_results, overall_rating)
        
        return {
            "criteria_matches": criteria_results,
            "overall_rating": overall_rating,
            "explanation": explanation
        }

# Global service instance
_o1a_service_instance = None

def assess_o1a_qualification(cv_text: str) -> Dict[str, Any]:
    """
    Assess a CV for O-1A visa qualification.
    
    Args:
        cv_text: Text content of the CV
        
    Returns:
        Assessment results
    """
    global _o1a_service_instance
    if _o1a_service_instance is None:
        _o1a_service_instance = O1AAssessmentService()
    return _o1a_service_instance.assess_cv(cv_text) 