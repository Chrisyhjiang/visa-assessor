from typing import Dict, List, Any, Optional
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
            if direct_awards:
                logger.info(f"Directly extracted {len(direct_awards)} awards from CV")
                return {
                    "criterion": criterion,
                    "evidence": direct_awards,
                    "confidence": 0.9  # High confidence for directly extracted awards
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
        
        prompt = f"""
        You are an expert immigration consultant specializing in O-1A visas. Your task is to analyze a CV for evidence of the {criterion} criterion.
        
        CRITERION INFORMATION:
        {criterion_info}
        
        WHAT WE'RE LOOKING FOR:
        {criterion_description}
        
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
        
        # If "No evidence found" in the response, clear any evidence and set confidence to 0
        if "No evidence found" in response:
            evidence = []
            confidence = 0.0
        
        # If we have pre-categorized sentences but no evidence from the LLM, use those instead
        if not evidence and categorized_sentences:
            evidence = categorized_sentences[:5]  # Use up to 5 pre-categorized sentences
            confidence = 0.7  # Assign a reasonable confidence score
        
        # Log the results for debugging
        logger.info(f"Criterion: {criterion}, Evidence count: {len(evidence)}, Confidence: {confidence}")
        
        return {
            "criterion": criterion,
            "evidence": evidence,
            "confidence": confidence
        }
    
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
            Look for ANY mentions of:
            - Leadership roles at prestigious organizations in ANY industry
            - Senior positions, executive titles, or management roles
            - Management of teams, departments, or organizations
            - Critical roles in significant projects or initiatives
            - Employment at well-known, respected companies or institutions
            - Roles requiring specialized expertise or rare skills
            - Promotions to positions of greater responsibility
            - Founding or co-founding companies or organizations
            - C-suite positions or equivalent executive roles
            - Director-level positions or department leadership
            - Key positions in high-profile projects or programs
            - Roles with significant decision-making authority or influence
            """,
            
            "High_remuneration": """
            Look for ANY mentions of:
            - Salary, compensation, remuneration figures in ANY industry
            - Bonuses, equity, stock options, or performance-based compensation
            - Financial success indicators or earnings information
            - Funding secured, grants obtained, or capital raised
            - Revenue generated or managed for organizations
            - Budget responsibility or financial oversight
            - Compensation significantly above industry average
            - Management of substantial financial resources or assets
            - Profit-sharing, royalties, or commission-based earnings
            - Contract values, deal sizes, or transaction amounts
            - Valuation of companies founded or led
            - Financial impact of work or contributions
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
        # Count criteria with evidence
        criteria_with_evidence = sum(1 for result in criteria_results.values() if result["evidence"])
        
        # Calculate average confidence across criteria with evidence
        confidences = [result["confidence"] for result in criteria_results.values() if result["evidence"]]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        logger.info(f"Criteria with evidence: {criteria_with_evidence}, Average confidence: {avg_confidence}")
        
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
        
        criteria_results = {}
        
        # Process each criterion
        for criterion in self.criteria:
            # Get criterion information from the knowledge base
            kb_results = self.rag_service.query_knowledge_base(f"Detailed explanation of {criterion} criterion for O-1A visa with examples", top_k=2)
            criterion_info = kb_results[0]["content"] if kb_results else ""
            
            # Analyze the CV for this criterion
            result = self._analyze_criterion(processed_cv, criterion, criterion_info)
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
        awards_section = self._extract_section(cv_text, ["AWARDS", "HONORS", "RECOGNITIONS", "PRIZES"])
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
                        awards.append(award)
                # Lines that mention "award", "prize", "medal", etc. are likely awards
                elif any(keyword in line.lower() for keyword in ["award", "prize", "medal", "honor", "fellowship", "scholarship", "grant", "recognition"]):
                    if len(line) > 5:  # Ensure it's not just a short marker
                        awards.append(line)
        
        # Also look for award-like patterns throughout the CV
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
                    if award and len(award) > 5:  # Ensure it's not just a short marker
                        awards.append(award)
        
        return awards
    
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