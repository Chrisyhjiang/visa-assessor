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
    """Service to assess O-1A visa qualification based on CV"""
    
    def __init__(self):
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
        
        Args:
            cv_content: Raw bytes of the CV file
            file_extension: File extension (pdf, docx, txt)
            
        Returns:
            Dict containing assessment results
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