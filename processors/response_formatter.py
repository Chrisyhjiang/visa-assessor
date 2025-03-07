import json
import ftfy
import re

def format_llm_response(llm_response: str) -> dict:
    """
    Format and clean the LLM response to ensure it's valid JSON
    """
    # Clean the text with ftfy
    cleaned_text = ftfy.fix_text(llm_response)
    
    # Extract JSON from the response (in case there's additional text)
    json_match = re.search(r'({[\s\S]*})', cleaned_text)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = cleaned_text
    
    try:
        # Parse the JSON
        response_dict = json.loads(json_str)
        
        # Ensure all required fields are present
        required_fields = ["criteria_matches", "qualification_rating", "rating_justification"]
        for field in required_fields:
            if field not in response_dict:
                response_dict[field] = "Not provided by model"
        
        # Ensure all criteria are present in criteria_matches
        all_criteria = [
            "awards", "membership", "press", "judging", 
            "original_contribution", "scholarly_articles", 
            "critical_employment", "high_remuneration"
        ]
        
        if "criteria_matches" in response_dict:
            for criterion in all_criteria:
                if criterion not in response_dict["criteria_matches"]:
                    response_dict["criteria_matches"][criterion] = []
        
        # Calculate criteria met if not provided
        if "criteria_met_count" not in response_dict:
            criteria_met = sum(
                1 for criterion in all_criteria 
                if response_dict["criteria_matches"].get(criterion) and len(response_dict["criteria_matches"][criterion]) > 0
            )
            response_dict["criteria_met_count"] = criteria_met
        
        return response_dict
        
    except json.JSONDecodeError:
        # Fallback for invalid JSON
        return {
            "error": "Failed to parse model response",
            "raw_response": cleaned_text,
            "criteria_matches": {criterion: [] for criterion in all_criteria},
            "qualification_rating": "Unknown",
            "rating_justification": "Could not determine due to parsing error",
            "criteria_met_count": 0
        } 