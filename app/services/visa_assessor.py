"""
Visa Assessor Service

This module provides the core functionality for assessing O-1A visa qualifications
based on CV content. It uses OpenAI's GPT models to analyze the text and determine
how well the applicant meets each of the O-1A visa criteria.

The assessment considers eight main criteria for O-1A visa qualification:
1. Awards and recognition
2. Membership in associations requiring outstanding achievement
3. Published material about the person
4. Participation as a judge of others' work
5. Original contributions of major significance
6. Authorship of scholarly articles
7. Critical employment in distinguished organizations
8. High salary or remuneration

The service provides both quantitative scores and qualitative evidence for each criterion.
"""

import io
import re
from typing import List, Dict
from openai import OpenAI
import os
from dotenv import load_dotenv
from app.utils.text_extraction import extract_text

# Force reload of environment variables
load_dotenv(override=True)

class VisaAssessor:
    """
    Service to assess O-1A visa qualification based on CV content.
    
    This class handles the interaction with OpenAI's API to analyze CV content
    and determine qualification for O-1A visa criteria. It provides both
    numerical scores and textual evidence for each criterion.
    
    Attributes:
        client (OpenAI): OpenAI API client instance
    """
    
    def __init__(self):
        """
        Initialize the VisaAssessor with OpenAI API credentials.
        
        Raises:
            ValueError: If OPENAI_API_KEY is not found in environment variables
        """
        # Debug: Print all possible places where OPENAI_API_KEY might be set
        env_file_key = os.getenv("OPENAI_API_KEY")
        env_var_key = os.environ.get("OPENAI_API_KEY")
        
        print("Debug API Key sources:")
        print(f"From .env file: {env_file_key[:10]}..." if env_file_key else "Not found in .env")
        print(f"From environment: {env_var_key[:10]}..." if env_var_key else "Not found in environment")
        
        if not env_file_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
            
        # Force use of the key from .env file
        self.client = OpenAI(api_key=env_file_key)
    
    def assess_cv(self, cv_content: bytes, file_extension: str) -> Dict:
        """
        Assess a CV for O-1A visa qualification using GPT-4.
        
        This method processes the CV content, extracts relevant information,
        and uses GPT-4 to analyze how well the applicant meets each O-1A
        visa criterion.
        
        Args:
            cv_content (bytes): Raw bytes of the CV file
            file_extension (str): File extension (pdf, docx, txt)
            
        Returns:
            Dict: Assessment results containing:
                - qualification_rating: Overall rating (high/medium/low)
                - overall_score: Aggregate score (0-1)
                - criteria_matches: Dict of criterion scores and evidence
                - explanation: Detailed explanation of assessment
                - recommendations: List of improvement suggestions
                - agent_explanation: USCIS officer-style explanation
                - agent_recommendations: Officer-style recommendations
                
        Raises:
            Exception: If CV processing or assessment fails
        """
        try:
            # Extract text from CV
            text = extract_text(cv_content, file_extension)
            
            # Clean up the text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Prepare the prompt for GPT-4
            prompt = f"""
            Analyze the following CV content and assess the applicant's qualification for an O-1A visa.
            The O-1A visa requires demonstrating extraordinary ability in sciences, education, business, or athletics.
            
            CV Content:
            {text}
            
            Please provide:
            1. A qualification rating (high, medium, low)
            2. An overall score (0-1)
            3. Evidence for each of the following criteria:
               - Awards and recognition
               - Membership in associations requiring outstanding achievement
               - Published material about the person
               - Participation as a judge of others' work
               - Original contributions of major significance
               - Authorship of scholarly articles
               - Critical employment in distinguished organizations
               - High salary or remuneration
            
            Format your response as a JSON object with the following structure:
            {{
                "qualification_rating": "high/medium/low",
                "overall_score": 0.XX,
                "criteria_matches": {{
                    "criterion_name": {{
                        "score": 0.XX,
                        "evidence": ["evidence1", "evidence2"]
                    }}
                }},
                "explanation": "brief explanation of the assessment",
                "recommendations": ["recommendation1", "recommendation2"]
            }}
            
            Make sure criteria_matches is a dictionary where keys are criterion names and values contain score and evidence list.
            """
            
            # Make the API call to GPT-4
            response = self.client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are an expert in O-1A visa assessment. Return responses in the exact JSON format specified."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            # Parse the response
            assessment = response.choices[0].message.content
            
            # Convert the string response to a dictionary
            try:
                import json
                result = json.loads(assessment)
                
                # Convert list format to dictionary format if needed
                if isinstance(result.get('criteria_matches'), list):
                    criteria_dict = {}
                    for item in result['criteria_matches']:
                        criteria_dict[item['criterion']] = {
                            'score': item['score'],
                            'evidence': item.get('matches', [])  # Handle both 'matches' and 'evidence' keys
                        }
                    result['criteria_matches'] = criteria_dict
                
                # Add agent_explanation and agent_recommendations using OpenAI
                evidence_text = "\n".join(
                    f"- {item}" for match in result["criteria_matches"].values() for item in match["evidence"]
                )
                agent_prompt = f"You are a USCIS visa officer. Given the following evidence from an O-1A visa assessment applicant's CV, write a professional, friendly explanation of their strengths and weaknesses, and provide 3-5 actionable recommendations for improvement.\n\nEvidence:\n{evidence_text}\n\nRespond in JSON with keys 'agent_explanation' and 'agent_recommendations' (a list)."
                agent_response = self.client.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=[
                        {"role": "system", "content": "You are an expert in O-1A visa assessment. Respond in the exact JSON format specified."},
                        {"role": "user", "content": agent_prompt}
                    ],
                    temperature=0.4
                )
                try:
                    agent_json = json.loads(agent_response.choices[0].message.content)
                    result["agent_explanation"] = agent_json.get("agent_explanation", "")
                    result["agent_recommendations"] = agent_json.get("agent_recommendations", [])
                except Exception:
                    result["agent_explanation"] = "(Agent response unavailable)"
                    result["agent_recommendations"] = []
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, return a structured error
                return {
                    "qualification_rating": "error",
                    "overall_score": 0.0,
                    "criteria_matches": {},
                    "explanation": "Error processing assessment",
                    "recommendations": ["Please try again with a different CV"]
                }
        except Exception as e:
            raise Exception(f"Error processing CV: {str(e)}") 