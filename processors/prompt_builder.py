def build_o1a_assessment_prompt(cv_text: str) -> str:
    """
    Build a prompt for the LLM to assess O-1A visa qualification
    """
    prompt = f"""You are an expert immigration consultant specializing in O-1A visa assessments. 
The O-1A visa is for individuals with extraordinary ability in sciences, education, business, or athletics.

Below are the 8 criteria for O-1A qualification. An applicant must meet at least 3 of these criteria:

1. Awards: Receipt of nationally or internationally recognized prizes/awards for excellence
2. Membership: Membership in associations requiring outstanding achievement
3. Press: Published material in professional/major trade publications or major media
4. Judging: Participation as a judge of others' work
5. Original Contribution: Original scientific, scholarly, or business-related contributions of major significance
6. Scholarly Articles: Authorship of scholarly articles in professional journals or major media
7. Critical Employment: Employment in a critical or essential capacity at distinguished organizations
8. High Remuneration: Command of a high salary or remuneration

TASK:
Analyze the following CV and:
1. Identify specific elements that satisfy each of the 8 O-1A criteria
2. For each identified element, explain why it qualifies
3. Provide an overall assessment rating (Low, Medium, or High) of the applicant's chances for O-1A qualification
4. Justify your rating

CV TEXT:
{cv_text}

Format your response as a JSON object with the following structure:
{{
  "criteria_matches": {{
    "awards": [list of specific matches with explanations],
    "membership": [list of specific matches with explanations],
    "press": [list of specific matches with explanations],
    "judging": [list of specific matches with explanations],
    "original_contribution": [list of specific matches with explanations],
    "scholarly_articles": [list of specific matches with explanations],
    "critical_employment": [list of specific matches with explanations],
    "high_remuneration": [list of specific matches with explanations]
  }},
  "qualification_rating": "Low|Medium|High",
  "rating_justification": "detailed explanation",
  "criteria_met_count": number of criteria met
}}
"""
    return prompt

def build_o1a_assessment_prompt_with_rag(cv_text: str, legal_context: str) -> str:
    """
    Build a prompt for the LLM to assess O-1A visa qualification with RAG context
    """
    prompt = f"""You are an expert immigration consultant specializing in O-1A visa assessments. 
The O-1A visa is for individuals with extraordinary ability in sciences, education, business, or athletics.

Below are relevant legal guidelines for O-1A visa qualification:

{legal_context}

TASK:
Analyze the following CV and:
1. Identify specific elements that satisfy each of the 8 O-1A criteria
2. For each identified element, explain why it qualifies
3. Provide an overall assessment rating (Low, Medium, or High) of the applicant's chances for O-1A qualification
4. Justify your rating

CV TEXT:
{cv_text}

Format your response as a JSON object with the following structure:
{{
  "criteria_matches": {{
    "awards": [list of specific matches with explanations],
    "membership": [list of specific matches with explanations],
    "press": [list of specific matches with explanations],
    "judging": [list of specific matches with explanations],
    "original_contribution": [list of specific matches with explanations],
    "scholarly_articles": [list of specific matches with explanations],
    "critical_employment": [list of specific matches with explanations],
    "high_remuneration": [list of specific matches with explanations]
  }},
  "qualification_rating": "Low|Medium|High",
  "rating_justification": "detailed explanation",
  "criteria_met_count": number of criteria met
}}
"""
    return prompt 