from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
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
            logger.info("Initializing Qwen2.5-0.5B model...")
            try:
                self.model_name = "Qwen/Qwen2.5-0.5B"
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
                logger.info("Qwen2.5-0.5B model initialized successfully")
            except Exception as e:
                # Fallback to a more commonly available model if Qwen fails
                logger.warning(f"Failed to load Qwen2.5-0.5B model: {str(e)}. Falling back to gpt2.")
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
                temperature=0.7,  # Slightly lower temperature for more focused outputs
                top_p=0.95,
                repetition_penalty=1.2,  # Penalize repetition
                no_repeat_ngram_size=3,  # Prevent repeating 3-grams
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,  # Set pad token explicitly
            )
        
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response
    
    def _analyze_criterion(self, cv_data: Dict, criterion: str, criterion_info: str) -> Dict[str, Any]:
        """
        Analyze a CV for a specific O-1A criterion.
        
        Args:
            cv_data: Processed CV data
            criterion: The criterion to analyze
            criterion_info: Information about the criterion
            
        Returns:
            Analysis results for the criterion
        """
        # Extract the full text and other data
        if isinstance(cv_data, dict) and "full_text" in cv_data:
            cv_text = cv_data["full_text"]
            categorized_sentences = cv_data.get("categorized_sentences", {}).get(criterion, [])
            entities = cv_data.get("entities", {})
        else:
            cv_text = cv_data
            categorized_sentences = []
            entities = {}
        
        # Special handling for Awards criterion
        if criterion == "Awards":
            # Try to extract awards directly from the CV
            direct_awards = self._extract_awards_directly(cv_text)
            # Filter out section titles and ensure we have actual awards
            direct_awards = [award for award in direct_awards if not self._is_section_title(award)]
            if direct_awards:
                logger.info(f"Directly extracted {len(direct_awards)} awards from CV")
                return {
                    "criterion": criterion,
                    "evidence": direct_awards,
                    "confidence": 0.9  # High confidence for directly extracted awards
                }
        
        # Special handling for High_remuneration criterion
        if criterion == "High_remuneration":
            # Try to extract salary information directly
            salary_info, salary_level = self._extract_salary_information(cv_text)
            # Filter out section titles
            salary_info = [info for info in salary_info if not self._is_section_title(info)]
            if salary_info:
                logger.info(f"Directly extracted salary information: {salary_level}")
                return {
                    "criterion": criterion,
                    "evidence": salary_info,
                    "confidence": self._get_salary_confidence(salary_level)
                }
        
        # Special handling for Critical_employment criterion
        if criterion == "Critical_employment":
            # Try to extract critical employment information directly
            critical_jobs = self._extract_critical_employment(cv_text)
            # Filter out section titles
            critical_jobs = [job for job in critical_jobs if not self._is_section_title(job)]
            if critical_jobs:
                logger.info(f"Directly extracted critical employment: {len(critical_jobs)} positions")
                return {
                    "criterion": criterion,
                    "evidence": critical_jobs,
                    "confidence": 0.85  # High confidence for directly extracted critical employment
                }
        
        # Create examples based on the criterion
        examples = self._get_criterion_examples(criterion)
        
        # Create a list of all criteria for reference
        all_criteria = ", ".join(self.criteria)
        
        # Create a more flexible description of what we're looking for
        criterion_description = self._get_criterion_description(criterion)
        
        # Create a special section highlighting the pre-categorized sentences
        categorized_section = ""
        if categorized_sentences:
            categorized_section = f"\n\n===== PRE-CATEGORIZED {criterion.upper()} CONTENT =====\n"
            for i, sentence in enumerate(categorized_sentences[:10]):  # Limit to 10 items to avoid overwhelming the model
                categorized_section += f"{i+1}. {sentence}\n"
        
        # Create a section for relevant named entities
        entities_section = ""
        if entities:
            relevant_entities = self._get_relevant_entities_for_criterion(criterion, entities)
            if relevant_entities:
                entities_section = f"\n\n===== RELEVANT NAMED ENTITIES FOR {criterion.upper()} =====\n"
                for entity_type, entity_list in relevant_entities.items():
                    if entity_list:
                        entities_section += f"{entity_type}:\n"
                        for entity in entity_list[:5]:  # Limit to 5 entities per type
                            entities_section += f"- {entity}\n"
        
        # Add specific instructions for High_remuneration and Critical_employment
        specific_instructions = ""
        if criterion == "High_remuneration":
            specific_instructions = """
            SPECIFIC SALARY THRESHOLDS:
            - LOW: Less than $150,000 per year
            - MEDIUM: $150,000 to $350,000 per year
            - HIGH: More than $350,000 per year
            
            Only assign HIGH confidence (80-100) if there is clear evidence of compensation exceeding $350,000.
            Assign MEDIUM confidence (50-79) for compensation between $150,000 and $350,000.
            Assign LOW confidence (1-49) for compensation below $150,000.
            """
        elif criterion == "Critical_employment":
            specific_instructions = """
            CRITICAL EMPLOYMENT CATEGORIES:
            - Government positions (especially senior roles)
            - Military positions (especially leadership roles)
            - STEM jobs (Science, Technology, Engineering, Mathematics)
            - Senior leadership positions at prestigious organizations
            - Specialized roles requiring rare expertise
            
            Focus particularly on government, military, and STEM positions as these are considered especially critical.
            """
        elif criterion == "Awards":
            specific_instructions = """
            IMPORTANT FOR AWARDS:
            - Only include SPECIFIC awards, prizes, medals, or honors the person has received
            - Do NOT include section titles like "AWARDS SECTION" or "HONORS AND AWARDS"
            - Each piece of evidence should be an actual award with its name and possibly the year
            - Examples of valid evidence: "Best Paper Award, NeurIPS 2017" or "Employee of the Year, 2019"
            - If you only see section titles but no actual awards, report "No evidence found"
            """
        
        prompt = f"""
        You are an expert immigration consultant specializing in O-1A visas. Your task is to analyze a CV for evidence of the {criterion} criterion.
        
        CRITERION INFORMATION:
        {criterion_info}
        
        WHAT WE'RE LOOKING FOR:
        {criterion_description}
        
        {specific_instructions}
        
        {categorized_section}
        {entities_section}
        
        APPROACH:
        1. First, mentally categorize each section/sentence of the CV into one of these categories:
           - {all_criteria}
           - Irrelevant
        
        2. Use fuzzy matching - look for information that CONCEPTUALLY matches the {criterion} criterion, even if it doesn't use the exact same wording.
        
        3. Consider both direct and indirect evidence. For example:
           - Direct: Explicitly mentioned awards, memberships, etc.
           - Indirect: Roles, responsibilities, or achievements that imply meeting the criterion
        
        CV CONTENT:
        {cv_text}
        
        TASK:
        1. Identify ANY information in the CV that could reasonably satisfy the {criterion} criterion, even if it's not an exact match.
        2. Extract specific quotes or information from the CV that supports this criterion.
        3. Provide a confidence score (0-100) for how strongly this evidence satisfies the criterion.
        4. If no evidence is found, simply state "No evidence found" and set confidence to 0.
        
        IMPORTANT INSTRUCTIONS:
        - DO NOT use "N/A" or similar placeholders as evidence. If there is no evidence, state "No evidence found".
        - Only include ACTUAL TEXT from the CV as evidence.
        - Section headers alone (like "AWARDS SECTION") are NOT valid evidence - you must include the actual awards.
        - If you can't find specific evidence, set confidence to 0 and return an empty evidence list.
        
        EXAMPLES OF EVIDENCE FOR {criterion.upper()}:
        {examples}
        
        FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:
        Evidence:
        - [Quote or information from the CV]
        - [Another quote or information from the CV]
        - ...
        
        Confidence: [0-100]
        
        IMPORTANT: Be thorough and creative in finding relevant evidence, but only include information that's actually in the CV.
        """
        
        response = self._generate_response(prompt, max_length=800)
        logger.info(f"Generated response for {criterion}: {response[:100]}...")
        
        # Parse the response to extract evidence and confidence using improved parsing
        evidence = []
        confidence = 0
        
        # More robust evidence extraction
        if "Evidence:" in response:
            evidence_section = response.split("Evidence:")[1]
            if "Confidence:" in evidence_section:
                evidence_section = evidence_section.split("Confidence:")[0]
            evidence_section = evidence_section.strip()
            
            # Extract evidence items, filtering out any that look like code or examples
            raw_evidence = [item.strip().lstrip("- ") for item in evidence_section.split("\n") if item.strip() and not item.strip().startswith("Confidence:")]
            
            # Filter out items that look like code or examples
            for item in raw_evidence:
                # Skip items that look like code (contain {, }, [, ], function calls, etc.)
                if any(code_marker in item for code_marker in ["{", "}", "def ", "print(", "return ", "import ", "class ", "function", "```"]):
                    continue
                # Skip items that are too short
                if len(item) < 5:
                    continue
                # Skip items that are just variable names or code-like patterns
                if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', item):
                    continue
                # Skip "N/A", "Not Applicable", "None", etc.
                if item.lower() in ["n/a", "not applicable", "none", "no evidence", "no evidence found"]:
                    continue
                # Skip section headers without actual content
                if self._is_section_title(item):
                    continue
                # Add the valid evidence
                evidence.append(item)
        
        # More robust confidence extraction
        if "Confidence:" in response:
            confidence_text = response.split("Confidence:")[1].strip()
            # Extract the first number found in the confidence text
            confidence_match = re.search(r'\d+(\.\d+)?', confidence_text)
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(0)) / 100.0  # Normalize to 0-1
                except ValueError:
                    confidence = 0.0
        
        # If we found evidence but confidence is 0, assign a minimum confidence
        if evidence and confidence == 0:
            confidence = 0.5  # Default to medium confidence if evidence was found but no confidence score
        
        # If "No evidence found" in the response or evidence list is empty, clear any evidence and set confidence to 0
        if "No evidence found" in response or not evidence:
            evidence = []
            confidence = 0.0
        
        # Apply specific rules for High_remuneration
        if criterion == "High_remuneration" and evidence:
            # Try to extract salary information from the evidence
            salary_level = self._determine_salary_level_from_evidence(evidence)
            if salary_level:
                confidence = self._get_salary_confidence(salary_level)
        
        # Apply specific rules for Critical_employment
        if criterion == "Critical_employment" and evidence:
            # Check if the evidence contains critical employment categories
            is_critical = self._is_critical_employment(evidence)
            if not is_critical:
                confidence = min(confidence, 0.4)  # Cap confidence if not in critical categories
        
        # Log the results for debugging
        logger.info(f"Criterion: {criterion}, Evidence count: {len(evidence)}, Confidence: {confidence}")
        
        return {
            "criterion": criterion,
            "evidence": evidence,
            "confidence": confidence
        }
    
    def _extract_salary_information(self, cv_text: str) -> Tuple[List[str], str]:
        """
        Extract salary information from the CV text.
        
        Args:
            cv_text: CV text
            
        Returns:
            Tuple of (list of salary evidence, salary level)
        """
        salary_evidence = []
        salary_level = "low"
        
        # Look for salary/compensation section
        compensation_section = self._extract_section(cv_text, ["SALARY", "COMPENSATION", "REMUNERATION", "COMPENSATION", "INCOME"])
        
        # Regular expressions to find salary information
        salary_patterns = [
            r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:k|K|thousand)?',  # $100,000 or $100k
            r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:thousand|k|K)',  # 100,000 or 100k
            r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|M|m)',  # 1,000,000 or 1M
            r'annual\s+(?:salary|compensation|income)(?:\s+of)?\s+\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',  # annual salary of $100,000
            r'(?:salary|compensation|income)(?:\s+of)?\s+\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',  # salary of $100,000
            r'(?:earn|earning|earned|makes|made)\s+\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',  # earning $100,000
            r'(?:package|total|compensation)\s+(?:of|worth|valued)?\s+\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)'  # package worth $100,000
        ]
        
        # First check the compensation section if available
        if compensation_section:
            lines = compensation_section.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ["salary", "compensation", "remuneration", "income", "earning", "package", "$"]):
                    salary_evidence.append(line.strip())
        
        # If no compensation section or no evidence found, check the entire CV
        if not salary_evidence:
            lines = cv_text.split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in ["salary", "compensation", "remuneration", "income", "earning", "package", "$"]):
                    salary_evidence.append(line.strip())
        
        # Determine salary level based on evidence
        highest_amount = 0
        for evidence in salary_evidence:
            for pattern in salary_patterns:
                matches = re.findall(pattern, evidence, re.IGNORECASE)
                for match in matches:
                    # Remove commas and convert to float
                    amount_str = match.replace(',', '')
                    try:
                        amount = float(amount_str)
                        # Adjust for thousands/millions
                        if 'k' in evidence.lower() or 'thousand' in evidence.lower():
                            amount *= 1000
                        elif 'm' in evidence.lower() or 'million' in evidence.lower():
                            amount *= 1000000
                        
                        highest_amount = max(highest_amount, amount)
                    except ValueError:
                        continue
        
        # Determine salary level based on thresholds
        if highest_amount >= 350000:
            salary_level = "high"
        elif highest_amount >= 150000:
            salary_level = "medium"
        else:
            salary_level = "low"
        
        # If we found salary evidence but couldn't determine an amount, default to medium
        if salary_evidence and highest_amount == 0:
            # Check if there are dollar signs or mentions of high compensation
            for evidence in salary_evidence:
                if '$' in evidence or any(term in evidence.lower() for term in ['high', 'substantial', 'significant', 'competitive']):
                    salary_level = "medium"
                    break
        
        return salary_evidence, salary_level
    
    def _get_salary_confidence(self, salary_level: str) -> float:
        """
        Get confidence score based on salary level.
        
        Args:
            salary_level: Salary level (low, medium, high)
            
        Returns:
            Confidence score (0-1)
        """
        if salary_level == "high":
            return 0.9  # High confidence for high salary
        elif salary_level == "medium":
            return 0.7  # Medium confidence for medium salary
        else:
            return 0.3  # Low confidence for low salary
    
    def _determine_salary_level_from_evidence(self, evidence: List[str]) -> str:
        """
        Determine salary level from evidence.
        
        Args:
            evidence: List of evidence strings
            
        Returns:
            Salary level (low, medium, high)
        """
        # Regular expressions to find salary information
        salary_patterns = [
            r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:k|K|thousand)?',  # $100,000 or $100k
            r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:thousand|k|K)',  # 100,000 or 100k
            r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:million|M|m)',  # 1,000,000 or 1M
        ]
        
        highest_amount = 0
        for item in evidence:
            for pattern in salary_patterns:
                matches = re.findall(pattern, item, re.IGNORECASE)
                for match in matches:
                    # Remove commas and convert to float
                    amount_str = match.replace(',', '')
                    try:
                        amount = float(amount_str)
                        # Adjust for thousands/millions
                        if 'k' in item.lower() or 'thousand' in item.lower():
                            amount *= 1000
                        elif 'm' in item.lower() or 'million' in item.lower():
                            amount *= 1000000
                        
                        highest_amount = max(highest_amount, amount)
                    except ValueError:
                        continue
        
        # Determine salary level based on thresholds
        if highest_amount >= 350000:
            return "high"
        elif highest_amount >= 150000:
            return "medium"
        else:
            return "low"
    
    def _extract_critical_employment(self, cv_text: str) -> List[str]:
        """
        Extract critical employment information from the CV text.
        
        Args:
            cv_text: CV text
            
        Returns:
            List of critical employment evidence
        """
        critical_evidence = []
        
        # Look for employment/experience section
        employment_section = self._extract_section(cv_text, ["EMPLOYMENT", "EXPERIENCE", "PROFESSIONAL EXPERIENCE", "WORK EXPERIENCE"])
        
        # Define critical employment categories
        government_keywords = ["government", "federal", "state", "public service", "civil service", "agency", "department of", "ministry of"]
        military_keywords = ["military", "army", "navy", "air force", "marine", "defense", "defence", "armed forces", "national guard", "coast guard"]
        stem_keywords = ["scientist", "engineer", "researcher", "developer", "programmer", "technologist", "mathematician", "data scientist", 
                        "ai", "artificial intelligence", "machine learning", "computer science", "physics", "chemistry", "biology", 
                        "technology", "technical", "engineering", "research", "development", "r&d", "laboratory", "lab"]
        
        # Check employment section if available
        if employment_section:
            # Split into job entries - typically separated by blank lines or new job titles
            job_entries = []
            current_entry = []
            
            lines = employment_section.split('\n')
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                
                # Check if this is a new job title (all caps, or starts with a position title)
                if (line_stripped.isupper() and len(line_stripped) > 3) or \
                   (i > 0 and not lines[i-1].strip() and line_stripped and not line_stripped.startswith('-')):
                    # Save the previous entry if it exists
                    if current_entry:
                        job_entries.append('\n'.join(current_entry))
                        current_entry = []
                
                # Add the line to the current entry
                if line_stripped:
                    current_entry.append(line)
            
            # Add the last entry
            if current_entry:
                job_entries.append('\n'.join(current_entry))
            
            # If no job entries were found, try a simpler approach
            if not job_entries:
                job_entries = [employment_section]
            
            # Process each job entry
            for job_entry in job_entries:
                # Skip if it's an education entry, not employment
                if any(edu_term in job_entry.lower() for edu_term in ["ph.d", "phd", "master", "bachelor", "m.s.", "b.s.", "m.a.", "b.a.", "university", "college", "school"]) and \
                   not any(work_term in job_entry.lower() for work_term in ["work", "job", "employ", "position", "role", "career", "profession"]):
                    continue
                
                job_entry_lower = job_entry.lower()
                
                # Check for STEM positions
                is_stem = any(keyword in job_entry_lower for keyword in stem_keywords)
                
                # Check for government positions
                is_government = any(keyword in job_entry_lower for keyword in government_keywords)
                
                # Check for military positions
                is_military = any(keyword in job_entry_lower for keyword in military_keywords)
                
                # Check for senior leadership positions
                is_leadership = any(leadership_term in job_entry_lower for leadership_term in 
                                   ["senior", "lead", "chief", "head", "director", "manager", "executive", 
                                    "officer", "principal", "founder", "co-founder", "president", "vice president", 
                                    "vp", "ceo", "cto", "cfo", "cio", "cso"])
                
                # If this is a critical employment position, add it to the evidence
                if is_stem or is_government or is_military or is_leadership:
                    # Extract the job title and company if possible
                    lines = job_entry.split('\n')
                    job_title = lines[0].strip() if lines else ""
                    
                    # Add the job title as evidence
                    if job_title and not self._is_section_title(job_title):
                        critical_evidence.append(job_title)
                    
                    # Add key responsibilities or achievements
                    for line in lines[1:]:
                        line = line.strip()
                        if line.startswith('-') and len(line) > 5:
                            # This is likely a bullet point with a responsibility or achievement
                            if any(keyword in line.lower() for keyword in stem_keywords + ["lead", "manage", "direct", "oversee", "supervise", "responsible"]):
                                critical_evidence.append(line)
        
        # If no critical evidence found using the job entries approach, try a simpler line-by-line approach
        if not critical_evidence and employment_section:
            lines = employment_section.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                line_lower = line.lower()
                
                # Check for STEM positions
                if any(keyword in line_lower for keyword in stem_keywords):
                    critical_evidence.append(line)
                
                # Check for government positions
                elif any(keyword in line_lower for keyword in government_keywords):
                    critical_evidence.append(line)
                
                # Check for military positions
                elif any(keyword in line_lower for keyword in military_keywords):
                    critical_evidence.append(line)
                
                # Check for leadership positions
                elif any(leadership_term in line_lower for leadership_term in 
                        ["senior", "lead", "chief", "head", "director", "manager", "executive", 
                         "officer", "principal", "founder", "co-founder", "president", "vice president", 
                         "vp", "ceo", "cto", "cfo", "cio", "cso"]):
                    critical_evidence.append(line)
        
        # Remove duplicates while preserving order
        seen = set()
        critical_evidence = [x for x in critical_evidence if not (x in seen or seen.add(x))]
        
        # Filter out any education-related items that might have slipped through
        critical_evidence = [x for x in critical_evidence if not self._is_education_item(x)]
        
        # Filter out section titles
        critical_evidence = [x for x in critical_evidence if not self._is_section_title(x)]
        
        return critical_evidence
    
    def _is_education_item(self, text: str) -> bool:
        """
        Check if a text is related to education rather than employment.
        
        Args:
            text: Text to check
            
        Returns:
            True if the text is related to education, False otherwise
        """
        education_terms = ["university", "college", "school", "ph.d", "phd", "master", "bachelor", 
                          "m.s.", "b.s.", "m.a.", "b.a.", "degree", "diploma", "thesis", 
                          "dissertation", "graduate", "undergraduate", "student", "professor", 
                          "faculty", "academic", "academia", "study", "studied", "major", "minor"]
        
        # Check if the text contains education terms but not employment terms
        has_education_terms = any(term in text.lower() for term in education_terms)
        has_employment_terms = any(term in text.lower() for term in ["work", "job", "employ", "position", "role", "career", "profession", "company", "corporation", "firm", "industry"])
        
        # If it has education terms but not employment terms, it's likely an education item
        return has_education_terms and not has_employment_terms
    
    def _is_critical_employment(self, evidence: List[str]) -> bool:
        """
        Check if evidence contains critical employment categories.
        
        Args:
            evidence: List of evidence strings
            
        Returns:
            True if evidence contains critical employment, False otherwise
        """
        # Define critical employment categories
        government_keywords = ["government", "federal", "state", "public service", "civil service", "agency", "department of", "ministry of"]
        military_keywords = ["military", "army", "navy", "air force", "marine", "defense", "defence", "armed forces", "national guard", "coast guard"]
        stem_keywords = ["scientist", "engineer", "researcher", "developer", "programmer", "technologist", "mathematician", "data scientist", 
                        "ai", "artificial intelligence", "machine learning", "computer science", "physics", "chemistry", "biology", 
                        "technology", "technical", "engineering", "research", "development", "r&d", "laboratory", "lab"]
        
        for item in evidence:
            item_lower = item.lower()
            
            # Check for government positions
            if any(keyword in item_lower for keyword in government_keywords):
                return True
            
            # Check for military positions
            if any(keyword in item_lower for keyword in military_keywords):
                return True
            
            # Check for STEM positions
            if any(keyword in item_lower for keyword in stem_keywords):
                return True
        
        return False
    
    def _get_criterion_examples(self, criterion: str) -> str:
        """
        Get examples for a specific criterion to help guide the model.
        
        Args:
            criterion: The criterion to get examples for
            
        Returns:
            String with examples for the criterion
        """
        examples = {
            "Awards": """
            TECHNOLOGY/SCIENCE:
            - 'ACM Distinguished Researcher Award, 2022'
            - 'Best Paper Award, NeurIPS Conference, 2017'
            - 'Google Spotlight Award for exceptional contributions to the BERT project'
            
            BUSINESS/CORPORATE:
            - 'Ernst & Young Entrepreneur of the Year Award, Northeast Region, 2019'
            - 'Named to Forbes 30 Under 30 list in Finance category, 2020'
            - 'Received the Stevie Award for Sales Executive of the Year, 2018'
            
            ARTS/ENTERTAINMENT:
            - 'Grammy Award for Best New Artist, 2021'
            - 'Pulitzer Prize for Drama for "The Central Park Five", 2019'
            - 'Cannes Film Festival Jury Prize for "The Lighthouse", 2018'
            
            MEDICINE/HEALTHCARE:
            - 'American Medical Association Scientific Achievement Award, 2020'
            - 'Healthcare Innovator of the Year, Modern Healthcare Magazine, 2019'
            - 'NIH Director's Pioneer Award for innovative research in immunotherapy'
            """,
            
            "Membership": """
            TECHNOLOGY/SCIENCE:
            - 'Senior Member, Association for Computing Machinery (ACM)'
            - 'Fellow, Institute of Electrical and Electronics Engineers (IEEE)'
            - 'Board Member, Women in Machine Learning (WiML)'
            
            BUSINESS/CORPORATE:
            - 'Member, Young Presidents' Organization (YPO) since 2018'
            - 'Board of Directors, National Retail Federation, 2019-present'
            - 'Advisory Board Member, Harvard Business School Digital Initiative'
            
            MEDICINE/HEALTHCARE:
            - 'Fellow, American College of Surgeons'
            - 'Member, National Academy of Medicine'
            - 'Board Certified in Cardiology by the American Board of Internal Medicine'
            
            LAW/GOVERNMENT:
            - 'Member, American Bar Association Antitrust Law Section'
            - 'Fellow, American College of Trial Lawyers'
            - 'Commissioner, Securities and Exchange Commission, 2017-2020'
            """,
            
            "Press": """
            TECHNOLOGY/SCIENCE:
            - 'Featured in MIT Technology Review, "35 Innovators Under 35," 2020'
            - 'TED Talk: "The Future of AI in Healthcare," 2019 (over 2 million views)'
            - 'Profiled in Wired magazine's special issue on "AI Pioneers", June 2021'
            
            BUSINESS/CORPORATE:
            - 'Cover story in Fortune Magazine, "Reinventing Retail", March 2022'
            - 'Featured on CNBC's "Power Lunch" discussing market trends, 2021'
            - 'Profiled in Wall Street Journal's "C-Suite Strategies" column, 2020'
            
            ARTS/ENTERTAINMENT:
            - 'Featured in Variety's "Directors to Watch" special issue, 2019'
            - 'In-depth interview in Rolling Stone about creative process, May 2021'
            - 'Subject of PBS documentary "The Visionary", aired nationally in 2020'
            
            SPORTS/FITNESS:
            - 'Cover athlete for Sports Illustrated's "Fittest 50" issue, 2022'
            - 'Featured in ESPN's "E:60" profile series, December 2021'
            - 'Subject of The Players' Tribune long-form article, "My Journey", 2020'
            """,
            
            "Judging": """
            TECHNOLOGY/SCIENCE:
            - 'Program Committee Member, NeurIPS Conference (2018-present)'
            - 'Associate Editor, Journal of Machine Learning Research (2020-present)'
            - 'Grant Reviewer, National Science Foundation Computer Science Division'
            
            BUSINESS/CORPORATE:
            - 'Judge, TechCrunch Disrupt Startup Battlefield, 2019-2021'
            - 'Selection Committee Member, Fortune 40 Under 40, 2020'
            - 'Venture Capital Pitch Competition Judge, Harvard Business School'
            
            ARTS/ENTERTAINMENT:
            - 'Jury Member, Sundance Film Festival Documentary Competition, 2022'
            - 'Selection Committee, National Book Awards, Fiction Category, 2021'
            - 'Judge, Kennedy Center Emerging Artist Awards, 2019-2020'
            
            EDUCATION/ACADEMIA:
            - 'External Reviewer for Faculty Tenure Committees at MIT and Stanford'
            - 'Fellowship Selection Committee, Fulbright Scholar Program, 2018-2021'
            - 'Dissertation Committee Member for 12 PhD candidates, 2015-present'
            """,
            
            "Original_contribution": """
            TECHNOLOGY/SCIENCE:
            - 'Pioneered a novel approach to few-shot learning that reduced training data requirements by 60%'
            - 'Invented and patented a new algorithm for real-time object detection in autonomous vehicles'
            - 'Developed the widely-adopted BERT language model, transforming natural language processing'
            
            BUSINESS/CORPORATE:
            - 'Created a revolutionary supply chain optimization system that reduced costs by 35% industry-wide'
            - 'Developed a new business model for subscription-based luxury retail, now industry standard'
            - 'Founded the first carbon-neutral shipping company, establishing new sustainability benchmarks'
            
            MEDICINE/HEALTHCARE:
            - 'Developed a novel surgical technique for minimally invasive heart valve repair'
            - 'Created a breakthrough diagnostic tool for early detection of pancreatic cancer'
            - 'Invented a portable, low-cost dialysis device for use in developing countries'
            
            ARTS/DESIGN:
            - 'Pioneered a new filmmaking technique combining virtual reality and traditional cinematography'
            - 'Created an innovative architectural approach to sustainable urban housing'
            - 'Developed a revolutionary method for digital art authentication using blockchain'
            """,
            
            "Scholarly_articles": """
            TECHNOLOGY/SCIENCE:
            - 'Rodriguez, M., et al. "Efficient Few-Shot Learning for Medical Image Analysis." Nature Machine Intelligence, 2022'
            - 'Published 7 papers at top-tier conferences (NeurIPS, ICML, ACL) with over 2,000 citations'
            - 'Co-authored "Deep Learning Approaches for Natural Language Processing", a definitive textbook in the field'
            
            BUSINESS/ECONOMICS:
            - 'Johnson, S., et al. "Disruption Dynamics: A New Model for Industry Evolution." Harvard Business Review, 2021'
            - 'Published "Market Efficiency in Emerging Economies" in Journal of Finance, cited over 500 times'
            - 'Co-authored the widely-used textbook "Strategic Management in the Digital Age"'
            
            MEDICINE/HEALTHCARE:
            - 'Lead author of "Novel Immunotherapy Approaches in Treating Advanced Melanoma" in The New England Journal of Medicine'
            - 'Published 15 peer-reviewed articles in high-impact medical journals with cumulative impact factor over 200'
            - 'Co-authored clinical practice guidelines for the American Heart Association, used nationwide'
            
            LAW/POLICY:
            - 'Author of "Constitutional Implications of Digital Privacy" in Harvard Law Review, cited in Supreme Court cases'
            - 'Published comprehensive analysis of international trade agreements in Journal of World Trade'
            - 'Lead author of influential white paper on financial regulation that shaped Dodd-Frank legislation'
            """,
            
            "Critical_employment": """
            TECHNOLOGY/SCIENCE:
            - 'SENIOR AI RESEARCHER, TechVision AI, San Francisco, CA | 2019 - Present'
            - 'Lead a team of 12 researchers and engineers developing state-of-the-art NLP models'
            - 'RESEARCH SCIENTIST, Google AI, Mountain View, CA | 2015 - 2019'
            
            BUSINESS/CORPORATE:
            - 'CHIEF EXECUTIVE OFFICER, Global Retail Partners, New York, NY | 2018 - Present'
            - 'SENIOR VICE PRESIDENT OF OPERATIONS, Fortune 500 Manufacturing Company | 2015 - 2018'
            - 'MANAGING DIRECTOR, Goldman Sachs Investment Banking Division | 2010 - 2015'
            
            MEDICINE/HEALTHCARE:
            - 'CHIEF OF CARDIOTHORACIC SURGERY, Mayo Clinic, Rochester, MN | 2017 - Present'
            - 'DIRECTOR OF CLINICAL RESEARCH, Memorial Sloan Kettering Cancer Center | 2012 - 2017'
            - 'CHIEF MEDICAL OFFICER, Innovative Therapeutics (biotech startup) | 2019 - Present'
            
            ARTS/ENTERTAINMENT:
            - 'CREATIVE DIRECTOR, Universal Studios Animation Division | 2016 - Present'
            - 'EXECUTIVE PRODUCER, Emmy-winning documentary series "Human Planet" | 2018 - 2020'
            - 'ARTISTIC DIRECTOR, New York Philharmonic Orchestra | 2014 - 2019'
            """,
            
            "High_remuneration": """
            TECHNOLOGY/SCIENCE:
            - 'Current annual compensation package at TechVision AI: $385,000 (base salary + bonus + equity)'
            - 'Previous compensation at Google (2018): $310,000'
            - 'Secured $2.5M in research funding through competitive grants and industry partnerships'
            
            BUSINESS/CORPORATE:
            - 'Total annual compensation as CEO: $1.2M base salary plus performance bonus of up to 150%'
            - 'Equity package valued at approximately $4.5M based on current company valuation'
            - 'Managed annual departmental budget of $50M and team of 120 professionals'
            
            MEDICINE/HEALTHCARE:
            - 'Annual clinical practice revenue exceeding $2.2M, placing in top 5% of specialists nationwide'
            - 'Research grants totaling $3.7M from NIH and private foundations over past 5 years'
            - 'Consulting arrangements with pharmaceutical companies valued at $175,000 annually'
            
            LEGAL/FINANCE:
            - 'Partner compensation share of $1.8M annually at AmLaw 100 law firm'
            - 'Managed hedge fund with $500M assets under management, generating 18% returns'
            - 'Annual bonus structure providing 2% of deal value for transactions over $100M'
            """
        }
        
        return examples.get(criterion, "- [Example evidence 1]\n- [Example evidence 2]")
    
    def _get_criterion_description(self, criterion: str) -> str:
        """
        Get a more flexible description of what we're looking for in each criterion.
        
        Args:
            criterion: The criterion to get a description for
            
        Returns:
            A description of what to look for
        """
        descriptions = {
            "Awards": """
            Look for ANY mentions of:
            - Awards, prizes, medals, honors, recognitions, distinctions in ANY field
            - Competitions won or placed in (academic, professional, artistic, athletic, etc.)
            - Grants, fellowships, scholarships based on merit or achievement
            - Recognition from industry, academic, professional, or community organizations
            - Any form of public acknowledgment for excellence or achievement
            - Industry-specific awards (e.g., Oscars for film, Grammys for music, Pulitzer for journalism)
            - Professional awards (e.g., "Teacher of the Year," "Salesperson of the Year")
            - Lifetime achievement or career recognition awards
            - Inclusion in prestigious lists (e.g., "30 Under 30," "40 Under 40," "Top 100")
            - Honors or distinctions that indicate exceptional achievement in any field
            """,
            
            "Membership": """
            Look for ANY mentions of:
            - Membership in professional associations, societies, organizations in ANY field
            - Board positions, committee roles, advisory positions in any organization
            - Leadership roles in professional groups or industry associations
            - Invitations to join exclusive professional communities or societies
            - Affiliations with prestigious institutions or groups
            - Professional designations or certifications that require membership
            - Elected or appointed positions in professional organizations
            - Fellow status in academic or professional societies
            - Participation in exclusive industry groups or forums
            - Membership on boards of directors, advisory boards, or governing bodies
            - Roles in standards-setting organizations or regulatory bodies
            """,
            
            "Press": """
            Look for ANY mentions of:
            - Media coverage, interviews, features in publications of ANY type
            - Articles written about the person (not by them) in any media outlet
            - Appearances in TV, radio, podcasts, online media, or streaming platforms
            - Public speaking engagements, keynotes, invited talks in any context
            - Conference presentations, panel discussions, industry forums
            - Being quoted or referenced in media as an expert or authority
            - Social media recognition, large following, or influencer status
            - Profiles or features in industry publications or mainstream media
            - Press releases or media announcements about the person's work
            - Documentary features, biographical content, or extended media coverage
            - Podcast or webinar appearances as a guest expert
            """,
            
            "Judging": """
            Look for ANY mentions of:
            - Serving as a judge, reviewer, evaluator of others' work in ANY field
            - Peer review activities for journals, conferences, or publications
            - Selection committee participation for awards, grants, or competitions
            - Editorial roles for publications, journals, or media outlets
            - Mentorship or evaluation of junior professionals with assessment responsibilities
            - Grant review panels or funding decision committees
            - Competition judging in any industry or field
            - Serving on thesis or dissertation committees
            - Evaluating applications, submissions, or entries for selective programs
            - Participation in audition, casting, or talent selection processes
            - Quality assessment, accreditation, or certification activities
            """,
            
            "Original_contribution": """
            Look for ANY mentions of:
            - Innovations, inventions, novel approaches in ANY field
            - Patents, intellectual property, or proprietary methods
            - Development of new methods, techniques, or technologies
            - Research breakthroughs or significant findings
            - Creation of new products, services, systems, or business models
            - Improvements to existing processes that had significant impact
            - Pioneering work in a field, subfield, or industry
            - Founding companies, organizations, or initiatives
            - Creating artistic works, designs, or creative content with significant impact
            - Developing new theories, frameworks, or paradigms
            - Introducing novel approaches that changed industry practices
            - Establishing new standards, protocols, or best practices
            """,
            
            "Scholarly_articles": """
            Look for ANY mentions of:
            - Publications in journals, conferences, books in ANY field
            - Research papers, articles, book chapters, or scholarly contributions
            - Citation counts, h-index, impact factors, or publication metrics
            - Authorship or co-authorship of academic or professional works
            - Technical reports, white papers, industry publications
            - Published research findings or scholarly analysis
            - Dissertation or thesis publications
            - Books, textbooks, or educational materials authored
            - Peer-reviewed publications or juried submissions
            - Industry reports, market analyses, or professional publications
            - Digital publications, online journals, or web-based scholarly content
            - Contributions to edited volumes, handbooks, or reference works
            """,
            
            "Critical_employment": """
            Look SPECIFICALLY for these categories of employment:
            
            1. GOVERNMENT POSITIONS:
            - Federal, state, or local government roles
            - Public service positions
            - Civil service roles
            - Regulatory agency positions
            - Policy-making roles
            - Diplomatic positions
            
            2. MILITARY POSITIONS:
            - Any branch of armed forces
            - Defense or national security roles
            - Military research or development
            - Military leadership positions
            
            3. STEM JOBS (Science, Technology, Engineering, Mathematics):
            - Scientific research positions
            - Engineering roles
            - Technology development positions
            - Computer science and programming roles
            - Data science and analytics positions
            - Mathematics and statistical roles
            - Research and development positions
            - Laboratory or technical positions
            
            4. SENIOR LEADERSHIP at prestigious organizations (secondary importance):
            - C-suite positions
            - Director-level roles
            - Department heads
            - Team leadership positions
            
            NOTE: Government, military, and STEM positions are considered MOST critical and should be given highest priority.
            """,
            
            "High_remuneration": """
            Look for ANY mentions of compensation with SPECIFIC ATTENTION to the following salary thresholds:
            
            1. HIGH SALARY (most significant): $350,000+ per year
            - Annual compensation packages exceeding $350,000
            - Base salary plus bonuses/equity totaling over $350,000
            - Income or earnings statements showing $350,000+
            - Contracts or offers with compensation above $350,000
            
            2. MEDIUM SALARY: $150,000 to $350,000 per year
            - Annual compensation between $150,000 and $350,000
            - Base salary plus benefits in this range
            - Income or earnings statements in this range
            
            3. LOW SALARY (not significant): Less than $150,000 per year
            - Annual compensation below $150,000
            
            Also look for:
            - Budget responsibility or financial oversight of significant resources
            - Management of substantial funds or assets
            - Revenue generation or profit responsibility
            - Equity, stock options, or ownership stakes with significant value
            - Funding secured or grants obtained (especially in millions)
            
            NOTE: Only compensation above $150,000 is considered significant, with $350,000+ being most impactful.
            """
        }
        
        return descriptions.get(criterion, "Look for any information related to this criterion, using flexible interpretation across all fields and industries.")
    
    def _determine_overall_rating(self, criteria_results: Dict[str, Dict[str, Any]]) -> QualificationRating:
        """
        Determine the overall O-1A qualification rating based on criteria results.
        
        Args:
            criteria_results: Results for each criterion
            
        Returns:
            Overall qualification rating
        """
        # Count criteria with evidence and reasonable confidence
        criteria_with_evidence = []
        for criterion, result in criteria_results.items():
            # Only count criteria with both evidence and confidence > 0.3
            if result["evidence"] and result["confidence"] > 0.3:
                criteria_with_evidence.append(criterion)
        
        # Calculate average confidence across criteria with evidence
        confidences = [result["confidence"] for criterion, result in criteria_results.items() 
                      if criterion in criteria_with_evidence]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Log detailed information for debugging
        logger.info(f"Criteria with evidence: {len(criteria_with_evidence)}")
        logger.info(f"Criteria: {criteria_with_evidence}")
        logger.info(f"Average confidence: {avg_confidence}")
        logger.info(f"Individual confidences: {confidences}")
        
        # O-1A requires meeting at least 3 criteria
        if len(criteria_with_evidence) >= 3 and avg_confidence > 0.7:
            logger.info("Assigning HIGH rating (3+ criteria with high confidence)")
            return QualificationRating.HIGH
        elif len(criteria_with_evidence) >= 3 and avg_confidence > 0.4:
            logger.info("Assigning MEDIUM rating (3+ criteria with medium confidence)")
            return QualificationRating.MEDIUM
        elif len(criteria_with_evidence) >= 2:
            logger.info("Assigning LOW rating (2 criteria with evidence)")
            return QualificationRating.LOW
        else:
            logger.info("Assigning LOW rating (insufficient criteria)")
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
        # Only consider criteria with actual evidence and reasonable confidence
        criteria_met = [criterion for criterion, result in criteria_results.items() 
                       if result["evidence"] and result["confidence"] > 0.3]
        
        # All criteria not in criteria_met are considered not met
        criteria_not_met = [criterion for criterion in self.criteria if criterion not in criteria_met]
        
        # Create a summary of evidence for met criteria
        evidence_summary = ""
        for criterion in criteria_met:
            result = criteria_results[criterion]
            evidence_list = "\n".join([f"  - {item}" for item in result["evidence"][:3]])  # Limit to top 3 pieces of evidence
            evidence_summary += f"\n\n{criterion} (Confidence: {result['confidence']:.2f}):\n{evidence_list}"
        
        # Create a more structured prompt for better explanations
        prompt = f"""
        You are an expert immigration consultant specializing in O-1A visas. Your task is to provide a clear, factual explanation of this O-1A visa qualification assessment.
        
        ASSESSMENT RESULTS:
        - Overall Rating: {overall_rating.value}
        - Criteria Met ({len(criteria_met)}): {', '.join(criteria_met) if criteria_met else 'None'}
        - Criteria Not Met ({len(criteria_not_met)}): {', '.join(criteria_not_met) if criteria_not_met else 'None'}
        
        EVIDENCE SUMMARY:{evidence_summary}
        
        TASK:
        Write a clear, concise explanation (3-5 sentences) of this O-1A visa qualification assessment. 
        
        Your explanation MUST:
        1. Be factual and directly based on the criteria met/not met
        2. Explain that an O-1A visa requires meeting at least 3 of the 8 criteria
        3. Explain why the applicant received their rating (low/medium/high)
        4. Avoid mentioning non-existent visa types or making up information
        5. Be professional and straightforward
        
        DO NOT:
        - Mention non-existent visa categories (like "O-4B" or "O-BiG")
        - Include placeholder text or incomplete sentences
        - Make up additional qualifications not in the evidence
        - Use overly complex language or jargon
        
        EXAMPLE OF A GOOD EXPLANATION:
        "Based on the assessment, the applicant meets 2 of the 8 criteria required for an O-1A visa: Awards and Critical Employment. Since an O-1A visa requires meeting at least 3 criteria, the applicant received a low rating. The applicant shows strong evidence in the areas of Awards and Critical Employment, but would need to demonstrate qualifications in at least one additional criterion to qualify for an O-1A visa."
        """
        
        explanation = self._generate_response(prompt, max_length=600)
        
        # Clean up the explanation to remove any potential hallucinations
        explanation = explanation.replace("Assistant:", "").replace("I'll provide", "").replace("Here's my", "")
        
        # Check for nonsensical visa types and other issues
        problematic_terms = ["O-4B", "O-BiG", "O-BM", "O-MUST", "O-Muste", "omitted information", "[", "]", "---"]
        for term in problematic_terms:
            if term in explanation:
                # If we detect problematic content, generate a simple fallback explanation
                if len(criteria_met) >= 3:
                    if overall_rating == QualificationRating.HIGH:
                        return f"The applicant meets {len(criteria_met)} of the 8 criteria required for an O-1A visa with high confidence: {', '.join(criteria_met)}. Since an O-1A visa requires meeting at least 3 criteria, the applicant received a high rating and appears to be well-qualified for an O-1A visa."
                    else:
                        return f"The applicant meets {len(criteria_met)} of the 8 criteria required for an O-1A visa: {', '.join(criteria_met)}. Since an O-1A visa requires meeting at least 3 criteria, the applicant received a medium rating and may qualify for an O-1A visa."
                else:
                    return f"The applicant meets {len(criteria_met)} of the 8 criteria required for an O-1A visa: {', '.join(criteria_met) if criteria_met else 'None'}. Since an O-1A visa requires meeting at least 3 criteria, the applicant received a low rating and does not currently qualify for an O-1A visa."
        
        return explanation
    
    def assess_cv(self, cv_text: str) -> Dict[str, Any]:
        """
        Assess a CV for O-1A visa qualification.
        
        Args:
            cv_text: Text content of the CV or parsed CV data
            
        Returns:
            Assessment results
        """
        # Check if cv_text is already parsed data or raw text
        if isinstance(cv_text, dict) and "full_text" in cv_text:
            parsed_cv = cv_text
            # Pre-process the CV to extract and highlight key sections
            processed_cv = self._preprocess_cv_for_analysis(parsed_cv)
        else:
            # Pre-process the CV to extract and highlight key sections
            processed_cv = self._preprocess_cv_for_analysis(cv_text)
        
        # Extract all evidence directly first
        direct_evidence = self._extract_all_direct_evidence(processed_cv)
        
        criteria_results = {}
        
        # Process each criterion
        for criterion in self.criteria:
            # Get criterion information from the knowledge base
            kb_results = self.rag_service.query_knowledge_base(f"Detailed explanation of {criterion} criterion for O-1A visa with examples", top_k=2)
            criterion_info = kb_results[0]["content"] if kb_results else ""
            
            # Check if we have direct evidence for this criterion
            if criterion in direct_evidence and direct_evidence[criterion]:
                logger.info(f"Using directly extracted evidence for {criterion}: {len(direct_evidence[criterion])} items")
                
                # Determine confidence based on criterion and evidence
                if criterion == "Awards":
                    confidence = 0.9  # High confidence for awards
                elif criterion == "High_remuneration":
                    # For High_remuneration, adjust confidence based on salary level
                    salary_level = self._determine_salary_level_from_evidence(direct_evidence[criterion])
                    confidence = self._get_salary_confidence(salary_level)
                elif criterion == "Critical_employment":
                    # For Critical_employment, check if it's actually critical
                    is_critical = self._is_critical_employment(direct_evidence[criterion])
                    confidence = 0.85 if is_critical else 0.4
                else:
                    confidence = 0.85  # Good confidence for other directly extracted evidence
                
                criteria_results[criterion] = {
                    "criterion": criterion,
                    "evidence": direct_evidence[criterion],
                    "confidence": confidence
                }
                
                # Log the result
                logger.info(f"Direct evidence for {criterion}: {len(direct_evidence[criterion])} items, confidence: {confidence}")
            else:
                # Analyze the CV for this criterion using the LLM
                result = self._analyze_criterion(processed_cv, criterion, criterion_info)
                criteria_results[criterion] = result
                logger.info(f"LLM analysis for {criterion}: {len(result['evidence'])} items, confidence: {result['confidence']}")
        
        # Determine overall rating
        overall_rating = self._determine_overall_rating(criteria_results)
        
        # Generate explanation
        explanation = self._generate_explanation(criteria_results, overall_rating)
        
        return {
            "criteria_matches": criteria_results,
            "overall_rating": overall_rating,
            "explanation": explanation
        }
    
    def _extract_all_direct_evidence(self, cv_data: Dict) -> Dict[str, List[str]]:
        """
        Extract evidence for all criteria directly from the CV.
        
        Args:
            cv_data: Processed CV data
            
        Returns:
            Dictionary mapping criteria to lists of evidence
        """
        if isinstance(cv_data, dict) and "full_text" in cv_data:
            cv_text = cv_data["full_text"]
        else:
            cv_text = cv_data
            
        direct_evidence = {criterion: [] for criterion in self.criteria}
        
        # Extract awards
        direct_evidence["Awards"] = self._extract_awards_directly(cv_text)
        
        # Extract critical employment
        direct_evidence["Critical_employment"] = self._extract_critical_employment(cv_text)
        
        # Extract high remuneration
        salary_info, _ = self._extract_salary_information(cv_text)
        direct_evidence["High_remuneration"] = salary_info
        
        # Extract memberships
        direct_evidence["Membership"] = self._extract_memberships_directly(cv_text)
        
        # Extract press coverage
        direct_evidence["Press"] = self._extract_press_directly(cv_text)
        
        # Extract judging activities
        direct_evidence["Judging"] = self._extract_judging_directly(cv_text)
        
        # Extract original contributions
        direct_evidence["Original_contribution"] = self._extract_original_contributions_directly(cv_text)
        
        # Extract scholarly articles
        direct_evidence["Scholarly_articles"] = self._extract_scholarly_articles_directly(cv_text)
        
        # Filter out section titles and ensure we have actual evidence
        for criterion in self.criteria:
            direct_evidence[criterion] = [item for item in direct_evidence[criterion] if not self._is_section_title(item)]
        
        # Log the extracted evidence for debugging
        for criterion, evidence in direct_evidence.items():
            logger.info(f"Directly extracted {len(evidence)} items for {criterion}")
            if evidence:
                logger.info(f"Sample evidence for {criterion}: {evidence[0]}")
        
        return direct_evidence
    
    def _extract_memberships_directly(self, cv_text: str) -> List[str]:
        """
        Extract memberships directly from the CV text.
        
        Args:
            cv_text: CV text
            
        Returns:
            List of extracted memberships
        """
        memberships = []
        
        # Look for the Memberships section
        memberships_section = self._extract_section(cv_text, ["MEMBERSHIP", "MEMBERSHIPS", "PROFESSIONAL MEMBERSHIPS", "AFFILIATIONS", "PROFESSIONAL AFFILIATIONS", "PROFESSIONAL MEMBERSHIPS & SERVICE"])
        
        if memberships_section:
            # Split the section into lines
            lines = memberships_section.split('\n')
            
            # Skip the header line(s)
            start_idx = 0
            for i, line in enumerate(lines):
                if "-----" in line or "=====" in line:
                    start_idx = i + 1
                    break
            
            # Process each line that looks like a membership
            for line in lines[start_idx:]:
                line = line.strip()
                # Skip empty lines or section dividers
                if not line or line.startswith("-----") or line.startswith("====="):
                    continue
                
                # Lines starting with - or • are likely membership items
                if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                    membership = line.lstrip("-•* ").strip()
                    if membership and len(membership) > 5:  # Ensure it's not just a short marker
                        memberships.append(membership)
                # Lines that mention "member", "fellow", etc. are likely memberships
                elif any(keyword in line.lower() for keyword in ["member", "fellow", "association", "society", "board", "committee", "council", "affiliate"]):
                    if len(line) > 5:  # Ensure it's not just a short marker
                        memberships.append(line)
        
        # If no memberships section or no memberships found, look for membership-like patterns throughout the CV
        if not memberships:
            lines = cv_text.split('\n')
            for line in lines:
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue
                
                # Check for lines that look like memberships
                if (line.startswith("-") or line.startswith("•") or line.startswith("*")) and any(keyword in line.lower() for keyword in ["member", "fellow", "association", "society", "board", "committee", "council", "affiliate"]):
                    membership = line.lstrip("-•* ").strip()
                    if membership and len(membership) > 5:  # Ensure it's not just a short marker
                        memberships.append(membership)
        
        return memberships
    
    def _extract_press_directly(self, cv_text: str) -> List[str]:
        """
        Extract press coverage directly from the CV text.
        
        Args:
            cv_text: CV text
            
        Returns:
            List of extracted press coverage
        """
        press_items = []
        
        # Look for the Press/Media section
        press_section = self._extract_section(cv_text, ["PRESS", "MEDIA", "MEDIA COVERAGE", "INTERVIEWS", "PUBLIC APPEARANCES", "INVITED TALKS", "TALKS", "PRESENTATIONS", "INVITED TALKS & MEDIA"])
        
        if press_section:
            # Split the section into lines
            lines = press_section.split('\n')
            
            # Skip the header line(s)
            start_idx = 0
            for i, line in enumerate(lines):
                if "-----" in line or "=====" in line:
                    start_idx = i + 1
                    break
            
            # Process each line that looks like press coverage
            for line in lines[start_idx:]:
                line = line.strip()
                # Skip empty lines or section dividers
                if not line or line.startswith("-----") or line.startswith("====="):
                    continue
                
                # Lines starting with - or • are likely press items
                if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                    press_item = line.lstrip("-•* ").strip()
                    if press_item and len(press_item) > 5:  # Ensure it's not just a short marker
                        press_items.append(press_item)
                # Lines that mention media, interviews, talks, etc.
                elif any(keyword in line.lower() for keyword in ["featured", "interview", "talk", "speaker", "keynote", "panelist", "media", "press", "news", "article", "magazine", "newspaper", "television", "radio", "podcast"]):
                    if len(line) > 5:  # Ensure it's not just a short marker
                        press_items.append(line)
        
        # If no press section or no press items found, look for press-like patterns throughout the CV
        if not press_items:
            lines = cv_text.split('\n')
            for line in lines:
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue
                
                # Check for lines that look like press coverage
                if (line.startswith("-") or line.startswith("•") or line.startswith("*")) and any(keyword in line.lower() for keyword in ["featured", "interview", "talk", "speaker", "keynote", "panelist", "media", "press", "news", "article", "magazine", "newspaper", "television", "radio", "podcast"]):
                    press_item = line.lstrip("-•* ").strip()
                    if press_item and len(press_item) > 5:  # Ensure it's not just a short marker
                        press_items.append(press_item)
        
        return press_items
    
    def _extract_judging_directly(self, cv_text: str) -> List[str]:
        """
        Extract judging activities directly from the CV text.
        
        Args:
            cv_text: CV text
            
        Returns:
            List of extracted judging activities
        """
        judging_items = []
        
        # Look for judging-related sections
        judging_section = self._extract_section(cv_text, ["JUDGING", "REVIEWER", "EDITORIAL", "REVIEW ACTIVITIES", "PROFESSIONAL SERVICE", "SERVICE", "COMMITTEE"])
        
        # If no dedicated judging section, check the memberships section
        if not judging_section:
            judging_section = self._extract_section(cv_text, ["MEMBERSHIP", "MEMBERSHIPS", "PROFESSIONAL MEMBERSHIPS", "AFFILIATIONS", "PROFESSIONAL AFFILIATIONS", "PROFESSIONAL MEMBERSHIPS & SERVICE"])
        
        if judging_section:
            # Split the section into lines
            lines = judging_section.split('\n')
            
            # Process each line that looks like judging activity
            for line in lines:
                line = line.strip()
                # Skip empty lines or section dividers
                if not line or line.startswith("-----") or line.startswith("====="):
                    continue
                
                # Lines starting with - or • that mention judging activities
                if (line.startswith("-") or line.startswith("•") or line.startswith("*")) and any(keyword in line.lower() for keyword in ["judge", "judging", "reviewer", "review", "editor", "editorial", "committee", "evaluation", "evaluator", "assess", "assessor", "referee", "program committee"]):
                    judging_item = line.lstrip("-•* ").strip()
                    if judging_item and len(judging_item) > 5:  # Ensure it's not just a short marker
                        judging_items.append(judging_item)
                # Other lines that strongly indicate judging activities
                elif any(phrase in line.lower() for phrase in ["program committee", "editorial board", "associate editor", "editor", "reviewer", "review committee", "selection committee", "evaluation committee"]):
                    if len(line) > 5:  # Ensure it's not just a short marker
                        judging_items.append(line)
        
        # Look for judging activities throughout the CV
        if not judging_items:
            lines = cv_text.split('\n')
            for line in lines:
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue
                
                # Check for lines that look like judging activities
                if (line.startswith("-") or line.startswith("•") or line.startswith("*")) and any(keyword in line.lower() for keyword in ["judge", "judging", "reviewer", "review", "editor", "editorial", "committee", "evaluation", "evaluator", "assess", "assessor", "referee", "program committee"]):
                    judging_item = line.lstrip("-•* ").strip()
                    if judging_item and len(judging_item) > 5:  # Ensure it's not just a short marker
                        judging_items.append(judging_item)
        
        return judging_items
    
    def _extract_original_contributions_directly(self, cv_text: str) -> List[str]:
        """
        Extract original contributions directly from the CV text.
        
        Args:
            cv_text: CV text
            
        Returns:
            List of extracted original contributions
        """
        contributions = []
        
        # Look for sections that might contain original contributions
        contribution_sections = []
        for section_name in ["ORIGINAL CONTRIBUTIONS", "INNOVATIONS", "PATENTS", "RESEARCH", "PROFESSIONAL EXPERIENCE", "EXPERIENCE", "PROJECTS"]:
            section = self._extract_section(cv_text, [section_name])
            if section:
                contribution_sections.append(section)
        
        for section in contribution_sections:
            # Split the section into lines
            lines = section.split('\n')
            
            # Process each line that looks like an original contribution
            for line in lines:
                line = line.strip()
                # Skip empty lines or section dividers
                if not line or line.startswith("-----") or line.startswith("====="):
                    continue
                
                # Lines starting with - or • that mention original contributions
                if (line.startswith("-") or line.startswith("•") or line.startswith("*")) and any(keyword in line.lower() for keyword in ["pioneer", "develop", "create", "invent", "discover", "novel", "new", "innovative", "original", "first", "patent", "breakthrough", "revolutionize", "transform", "lead", "spearhead", "design", "architect", "implement"]):
                    contribution = line.lstrip("-•* ").strip()
                    if contribution and len(contribution) > 10:  # Ensure it's a substantial contribution
                        contributions.append(contribution)
        
        # If no contributions found in dedicated sections, look throughout the CV
        if not contributions:
            lines = cv_text.split('\n')
            for line in lines:
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue
                
                # Check for lines that look like original contributions
                if (line.startswith("-") or line.startswith("•") or line.startswith("*")) and any(keyword in line.lower() for keyword in ["pioneer", "develop", "create", "invent", "discover", "novel", "new", "innovative", "original", "first", "patent", "breakthrough", "revolutionize", "transform", "lead", "spearhead", "design", "architect", "implement"]):
                    contribution = line.lstrip("-•* ").strip()
                    if contribution and len(contribution) > 10:  # Ensure it's a substantial contribution
                        contributions.append(contribution)
        
        return contributions
    
    def _extract_scholarly_articles_directly(self, cv_text: str) -> List[str]:
        """
        Extract scholarly articles directly from the CV text.
        
        Args:
            cv_text: CV text
            
        Returns:
            List of extracted scholarly articles
        """
        articles = []
        
        # Look for the Publications section
        publications_section = self._extract_section(cv_text, ["PUBLICATIONS", "SCHOLARLY ARTICLES", "PAPERS", "RESEARCH PAPERS", "JOURNAL ARTICLES", "PUBLICATIONS & PATENTS"])
        
        if publications_section:
            # Split the section into lines
            lines = publications_section.split('\n')
            
            # Skip the header line(s)
            start_idx = 0
            for i, line in enumerate(lines):
                if "-----" in line or "=====" in line:
                    start_idx = i + 1
                    break
            
            # Process each line that looks like a publication
            current_publication = ""
            for line in lines[start_idx:]:
                line = line.strip()
                # Skip empty lines or section dividers
                if not line or line.startswith("-----") or line.startswith("====="):
                    if current_publication:
                        articles.append(current_publication)
                        current_publication = ""
                    continue
                
                # Check if this is a new publication entry
                if line.startswith("-") or line.startswith("•") or line.startswith("*") or line.startswith("1.") or line.startswith("2.") or line.startswith("3.") or line.startswith("4.") or line.startswith("5."):
                    # Save the previous publication if it exists
                    if current_publication:
                        articles.append(current_publication)
                    
                    # Start a new publication
                    current_publication = line.lstrip("-•*123456789. ").strip()
                else:
                    # Continue the current publication
                    if current_publication:
                        current_publication += " " + line
            
            # Add the last publication if it exists
            if current_publication:
                articles.append(current_publication)
        
        # If no publications section or no publications found, look for publication-like patterns throughout the CV
        if not articles:
            lines = cv_text.split('\n')
            for line in lines:
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue
                
                # Check for lines that look like publications
                if (line.startswith("-") or line.startswith("•") or line.startswith("*")) and any(keyword in line.lower() for keyword in ["publish", "publication", "paper", "article", "journal", "conference", "proceedings", "book", "chapter"]):
                    article = line.lstrip("-•* ").strip()
                    if article and len(article) > 10:  # Ensure it's a substantial publication
                        articles.append(article)
                # Also check for lines that mention patents
                elif "patent" in line.lower() and len(line) > 10:
                    articles.append(line)
        
        return articles
    
    def _preprocess_cv_for_analysis(self, cv_text: str) -> Dict:
        """
        Pre-process the CV to extract and highlight key sections for better analysis.
        
        Args:
            cv_text: Original CV text or parsed CV data
            
        Returns:
            Processed CV data with highlighted sections
        """
        # Check if cv_text is already parsed data
        if isinstance(cv_text, dict) and "full_text" in cv_text:
            full_text = cv_text["full_text"]
            categorized_sentences = cv_text.get("categorized_sentences", {})
            entities = cv_text.get("entities", {})
        else:
            full_text = cv_text
            categorized_sentences = {}
            entities = {}
        
        # Create a dictionary to map section headers to criteria
        section_to_criterion = {
            "AWARDS": "Awards",
            "AWARDS & HONORS": "Awards",
            "HONORS": "Awards",
            "HONORS & AWARDS": "Awards",
            "RECOGNITIONS": "Awards",
            "PRIZES": "Awards",
            
            "MEMBERSHIP": "Membership",
            "MEMBERSHIPS": "Membership",
            "PROFESSIONAL MEMBERSHIPS": "Membership",
            "AFFILIATIONS": "Membership",
            "PROFESSIONAL AFFILIATIONS": "Membership",
            
            "PUBLICATIONS": "Scholarly_articles",
            "SCHOLARLY ARTICLES": "Scholarly_articles",
            "PAPERS": "Scholarly_articles",
            "RESEARCH PAPERS": "Scholarly_articles",
            "JOURNAL ARTICLES": "Scholarly_articles",
            
            "PRESS": "Press",
            "MEDIA": "Press",
            "MEDIA COVERAGE": "Press",
            "INTERVIEWS": "Press",
            "PUBLIC APPEARANCES": "Press",
            
            "JUDGING": "Judging",
            "REVIEWER": "Judging",
            "EDITORIAL": "Judging",
            "REVIEW ACTIVITIES": "Judging",
            
            "PATENTS": "Original_contribution",
            "INNOVATIONS": "Original_contribution",
            "INVENTIONS": "Original_contribution",
            "ORIGINAL CONTRIBUTIONS": "Original_contribution",
            
            "EMPLOYMENT": "Critical_employment",
            "PROFESSIONAL EXPERIENCE": "Critical_employment",
            "WORK EXPERIENCE": "Critical_employment",
            "EXPERIENCE": "Critical_employment",
            
            "SALARY": "High_remuneration",
            "COMPENSATION": "High_remuneration",
            "REMUNERATION": "High_remuneration",
        }
        
        # Split the CV into lines
        lines = full_text.split('\n')
        processed_lines = []
        
        # Track the current section
        current_section = None
        
        # Process each line
        for line in lines:
            # Check if this line is a section header
            upper_line = line.strip().upper()
            
            # Check if this line matches any of our section headers
            for section_header, criterion in section_to_criterion.items():
                if section_header in upper_line:
                    # Mark the start of a new section with a special tag
                    current_section = criterion
                    processed_lines.append(f"\n===== {criterion.upper()} SECTION START =====")
                    processed_lines.append(line)
                    break
            else:
                # If we're in a recognized section, add a marker to the line
                if current_section:
                    # Check if this line might be the end of the section (e.g., a new section header)
                    if line.strip() and line.strip()[0].isupper() and line.strip()[-1] not in '.:,;' and len(line.strip()) > 3:
                        if any(char.islower() for char in line.strip()):
                            # This might be a new section header
                            current_section = None
                            processed_lines.append(line)
                        else:
                            # Still in the current section
                            processed_lines.append(f"[{current_section}] {line}")
                    else:
                        # Still in the current section
                        processed_lines.append(f"[{current_section}] {line}")
                else:
                    # Not in a recognized section
                    processed_lines.append(line)
        
        # Join the processed lines back into a single string
        processed_text = '\n'.join(processed_lines)
        
        # Add a special section at the beginning to explicitly highlight key information
        key_sections = self._extract_key_sections(full_text)
        if key_sections:
            processed_text = "===== IMPORTANT CV SECTIONS =====\n" + key_sections + "\n\n" + processed_text
        
        # Add named entities section if available
        if entities:
            entity_section = self._format_entities_section(entities)
            if entity_section:
                processed_text = "===== NAMED ENTITIES =====\n" + entity_section + "\n\n" + processed_text
        
        # Return a dictionary with all the processed information
        return {
            "full_text": processed_text,
            "categorized_sentences": categorized_sentences,
            "entities": entities
        }
    
    def _format_entities_section(self, entities: Dict[str, List[str]]) -> str:
        """
        Format named entities into a readable section.
        
        Args:
            entities: Dictionary of named entities
            
        Returns:
            Formatted entities section
        """
        if not entities:
            return ""
        
        sections = []
        
        # Map entity types to criteria
        entity_type_to_criteria = {
            "ORG": ["Membership", "Critical_employment", "Press"],
            "PERSON": [],
            "MONEY": ["High_remuneration"],
            "PERCENT": ["Original_contribution"],
            "CARDINAL": ["High_remuneration", "Original_contribution"],
            "WORK_OF_ART": ["Scholarly_articles"],
            "EVENT": ["Press"]
        }
        
        # Format each entity type
        for entity_type, entity_list in entities.items():
            if not entity_list:
                continue
                
            # Get relevant criteria for this entity type
            relevant_criteria = entity_type_to_criteria.get(entity_type, [])
            criteria_str = f" (Relevant to: {', '.join(relevant_criteria)})" if relevant_criteria else ""
            
            # Add the entity section
            sections.append(f"{entity_type}{criteria_str}:")
            for entity in entity_list[:10]:  # Limit to 10 entities per type
                sections.append(f"- {entity}")
            sections.append("")
        
        return "\n".join(sections)
    
    def _extract_key_sections(self, cv_text: str) -> str:
        """
        Extract key sections from the CV for explicit highlighting.
        
        Args:
            cv_text: Original CV text
            
        Returns:
            String containing key extracted sections
        """
        key_sections = []
        
        # Extract Awards section
        awards_section = self._extract_section(cv_text, ["AWARDS", "HONORS", "AWARDS & HONORS", "HONORS & AWARDS"])
        if awards_section:
            key_sections.append("AWARDS AND HONORS SECTION:\n" + awards_section)
        
        # Extract Publications section
        publications_section = self._extract_section(cv_text, ["PUBLICATIONS", "SCHOLARLY ARTICLES", "PAPERS"])
        if publications_section:
            key_sections.append("PUBLICATIONS SECTION:\n" + publications_section)
        
        # Extract Memberships section
        memberships_section = self._extract_section(cv_text, ["MEMBERSHIP", "PROFESSIONAL MEMBERSHIPS", "AFFILIATIONS"])
        if memberships_section:
            key_sections.append("MEMBERSHIPS SECTION:\n" + memberships_section)
        
        # Extract Press section
        press_section = self._extract_section(cv_text, ["PRESS", "MEDIA", "INVITED TALKS", "KEYNOTE"])
        if press_section:
            key_sections.append("PRESS AND MEDIA SECTION:\n" + press_section)
        
        # Extract Compensation section
        compensation_section = self._extract_section(cv_text, ["COMPENSATION", "SALARY", "REMUNERATION"])
        if compensation_section:
            key_sections.append("COMPENSATION SECTION:\n" + compensation_section)
        
        return "\n\n".join(key_sections)
    
    def _extract_section(self, cv_text: str, section_headers: List[str]) -> str:
        """
        Extract a specific section from the CV.
        
        Args:
            cv_text: CV text
            section_headers: List of possible section headers to look for
            
        Returns:
            Extracted section text or empty string if not found
        """
        lines = cv_text.split('\n')
        section_content = []
        in_section = False
        
        for i, line in enumerate(lines):
            upper_line = line.strip().upper()
            
            # Check if this line is the start of our target section
            if not in_section:
                for header in section_headers:
                    if header in upper_line:
                        in_section = True
                        section_content.append(line)
                        break
            else:
                # Check if this line might be the start of a new section
                if line.strip() and line.strip()[0].isupper() and line.strip()[-1] not in '.:,;' and len(line.strip()) > 3:
                    # Check if it's likely a new section header
                    if i < len(lines) - 1 and (not lines[i+1].strip() or lines[i+1].strip()[0] == '-'):
                        # This is likely a new section header, so we're done with our current section
                        break
                
                # Add the line to our section content
                section_content.append(line)
        
        return '\n'.join(section_content)
    
    def _extract_awards_directly(self, cv_text: str) -> List[str]:
        """
        Extract awards directly from the CV text using pattern matching.
        
        Args:
            cv_text: CV text
            
        Returns:
            List of extracted awards
        """
        awards = []
        
        # Look for the Awards section
        awards_section = self._extract_section(cv_text, ["AWARDS", "HONORS", "AWARDS & HONORS", "HONORS & AWARDS"])
        
        if awards_section:
            # Split the section into lines
            lines = awards_section.split('\n')
            
            # Skip the header line(s)
            start_idx = 0
            for i, line in enumerate(lines):
                if "-----" in line or "=====" in line:
                    start_idx = i + 1
                    break
            
            # Process each line that looks like an award
            for line in lines[start_idx:]:
                line = line.strip()
                # Skip empty lines or section dividers
                if not line or line.startswith("-----") or line.startswith("====="):
                    continue
                
                # Lines starting with - or • are likely award items
                if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                    award = line.lstrip("-•* ").strip()
                    if award and len(award) > 5:  # Ensure it's not just a short marker
                        # Skip if it's a section title
                        if not self._is_section_title(award):
                            awards.append(award)
                # Lines that mention "award", "prize", "medal", etc. are likely awards
                elif any(keyword in line.lower() for keyword in ["award", "prize", "medal", "honor", "fellowship", "scholarship", "grant", "recognition"]):
                    if len(line) > 5:  # Ensure it's not just a short marker
                        # Skip if it's a section title
                        if not self._is_section_title(line):
                            awards.append(line)
        
        # If no awards section or no awards found, look for award-like patterns throughout the CV
        if not awards:
            lines = cv_text.split('\n')
            for line in lines:
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue
                
                # Check for lines that look like awards
                if (line.startswith("-") or line.startswith("•") or line.startswith("*")) and any(keyword in line.lower() for keyword in ["award", "prize", "medal", "honor", "fellowship", "scholarship", "grant", "recognition"]):
                    award = line.lstrip("-•* ").strip()
                    if award and len(award) > 5 and not self._is_section_title(award):  # Ensure it's not just a short marker or section title
                        awards.append(award)
                # Also check for lines that mention receiving an award
                elif any(pattern in line.lower() for pattern in ["received award", "awarded", "recipient of", "won the", "honored with", "recognized with"]):
                    if len(line) > 10 and not self._is_section_title(line):  # Longer lines that mention receiving awards
                        awards.append(line)
        
        # Validate awards - ensure they're actually awards and not other achievements
        validated_awards = []
        for award in awards:
            # Skip items that are likely not awards
            if any(non_award in award.lower() for non_award in ["published", "publication", "paper", "article", "journal", "conference", "collaborated", "developed", "implemented", "designed", "created"]):
                # These are likely publications or work achievements, not awards
                continue
                
            # Skip education degrees unless they explicitly mention honors
            if any(degree in award.lower() for degree in ["ph.d", "phd", "master", "bachelor", "m.s.", "b.s.", "m.a.", "b.a."]) and not any(honor in award.lower() for honor in ["honor", "distinction", "cum laude", "summa", "magna", "scholarship", "fellowship"]):
                continue
                
            validated_awards.append(award)
        
        return validated_awards
    
    def _get_relevant_entities_for_criterion(self, criterion: str, entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Get named entities relevant to a specific criterion.
        
        Args:
            criterion: The criterion to get entities for
            entities: Dictionary of all named entities
            
        Returns:
            Dictionary of relevant entities for the criterion
        """
        if not entities:
            return {}
        
        # Map criteria to relevant entity types
        criterion_to_entity_types = {
            "Awards": ["ORG", "EVENT", "DATE"],
            "Membership": ["ORG"],
            "Press": ["ORG", "WORK_OF_ART", "EVENT"],
            "Judging": ["ORG", "EVENT"],
            "Original_contribution": ["ORG", "PERCENT", "CARDINAL", "PRODUCT"],
            "Scholarly_articles": ["ORG", "WORK_OF_ART"],
            "Critical_employment": ["ORG", "DATE"],
            "High_remuneration": ["MONEY", "CARDINAL", "PERCENT"]
        }
        
        # Get relevant entity types for this criterion
        relevant_types = criterion_to_entity_types.get(criterion, [])
        
        # Filter entities to only include relevant types
        relevant_entities = {}
        for entity_type in relevant_types:
            if entity_type in entities and entities[entity_type]:
                relevant_entities[entity_type] = entities[entity_type]
        
        return relevant_entities
    
    def _is_section_title(self, text: str) -> bool:
        """
        Check if a text is likely a section title rather than actual content.
        
        Args:
            text: Text to check
            
        Returns:
            True if the text is likely a section title, False otherwise
        """
        # Check for common section title patterns
        section_patterns = [
            r'(?i)section',
            r'(?i)awards\s*(and|&)?\s*honors',
            r'(?i)honors\s*(and|&)?\s*awards',
            r'(?i)^awards$',
            r'(?i)^honors$',
            r'(?i)^publications$',
            r'(?i)^experience$',
            r'(?i)^education$',
            r'(?i)^skills$',
            r'(?i)^memberships?$',
            r'(?i)^affiliations?$',
            r'(?i)^professional\s+experience$',
            r'(?i)^work\s+experience$',
            r'(?i)^employment$',
            r'(?i)^compensation$',
            r'(?i)^salary$',
            r'(?i)^remuneration$',
            r'(?i)^press$',
            r'(?i)^media$',
            r'(?i)^judging$',
            r'(?i)^original\s+contributions?$',
            r'(?i)^scholarly\s+articles?$',
            r'(?i)^critical\s+employment$',
            r'(?i)^high\s+remuneration$'
        ]
        
        # Check if the text matches any section pattern
        for pattern in section_patterns:
            if re.search(pattern, text):
                return True
        
        # Check if the text is all uppercase or ends with a colon
        if text.isupper() or text.endswith(':'):
            return True
        
        # Check if the text is short and contains "section" or similar words
        if len(text.split()) <= 4 and any(word.lower() in text.lower() for word in ["section", "header", "title", "heading"]):
            return True
        
        return False

# Global service instance
_o1a_service_instance = None

def assess_o1a_qualification(cv_text: str) -> Dict[str, Any]:
    """
    Assess a CV for O-1A visa qualification.
    
    Args:
        cv_text: Text content of the CV or parsed CV data
        
    Returns:
        Assessment results
    """
    global _o1a_service_instance
    if _o1a_service_instance is None:
        _o1a_service_instance = O1AAssessmentService()
    return _o1a_service_instance.assess_cv(cv_text) 