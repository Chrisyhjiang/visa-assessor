import os
import re
from typing import Optional, Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the O-1A criteria categories
O1A_CRITERIA = [
    "Awards",
    "Membership",
    "Press",
    "Judging",
    "Original_contribution",
    "Scholarly_articles",
    "Critical_employment",
    "High_remuneration"
]

# Keywords associated with each criterion
CRITERIA_KEYWORDS = {
    "Awards": [
        # General Award Terms
        "award", "prize", "medal", "honor", "recognition", "distinction", "fellowship", 
        "scholarship", "grant", "winner", "won", "recipient", "received", "honored", 
        "recognized", "distinguished", "achievement", "gold", "silver", "bronze", "first place",
        
        # Award Qualifiers
        "prestigious", "international", "national", "global", "world", "industry", "professional",
        "peer", "juried", "competitive", "selective", "merit-based", "excellence", "outstanding",
        "exceptional", "exemplary", "notable", "significant", "major", "premier", "top",
        
        # Award Types by Field
        "lifetime achievement", "career achievement", "innovation", "leadership", "service",
        "teaching", "research", "artistic", "creative", "humanitarian", "community", "volunteer",
        "entrepreneurship", "business", "technical", "scientific", "academic", "professional"
    ],
    "Membership": [
        # General Membership Terms
        "member", "membership", "fellow", "association", "society", "organization", 
        "committee", "board", "council", "affiliate", "affiliated", "elected", "senior member",
        "professional body", "institute", "association", "chair", "president", "vice president",
        "secretary", "treasurer", "director", "advisor", "advisory",
        
        # Membership Qualifiers
        "distinguished", "honorary", "lifetime", "elected", "invited", "selected", "appointed",
        "certified", "accredited", "chartered", "licensed", "registered", "professional",
        
        # Membership Roles
        "founding", "charter", "board of directors", "trustee", "governor", "officer",
        "executive committee", "steering committee", "working group", "task force",
        "special interest group", "chapter", "division", "section", "branch"
    ],
    "Press": [
        # Media Coverage Terms
        "press", "media", "news", "article", "feature", "interview", "publication", 
        "magazine", "newspaper", "journal", "television", "radio", "podcast", "blog", 
        "featured in", "appeared in", "mentioned in", "profiled in", "covered by",
        
        # Speaking/Presentation Terms
        "keynote", "speaker", "talk", "presentation", "panel", "panelist", "TED", "invited",
        "plenary", "distinguished lecture", "guest lecture", "public speaking", "address",
        
        # Media Engagement Qualifiers
        "exclusive", "in-depth", "special", "spotlight", "highlight", "showcase", "profile",
        "cover story", "front page", "headline", "breaking news", "press release", "media kit",
        
        # Digital/Social Media
        "viral", "trending", "followers", "subscribers", "channel", "platform", "influencer",
        "content creator", "thought leader", "expert commentary", "opinion piece", "editorial"
    ],
    "Judging": [
        # Evaluation Roles
        "judge", "judging", "reviewer", "review", "editor", "editorial", "committee", 
        "evaluation", "evaluator", "assess", "assessor", "referee", "adjudicate", 
        "adjudicator", "selection committee", "program committee", "peer review",
        "journal reviewer", "conference reviewer", "grant reviewer",
        
        # Judging Activities
        "jury", "juror", "panel", "panelist", "critique", "criticism", "appraisal",
        "examination", "screening", "vetting", "shortlisting", "deliberation",
        
        # Judging Contexts
        "competition", "contest", "award", "grant", "fellowship", "scholarship", "admission",
        "application", "submission", "entry", "portfolio", "performance", "exhibition",
        "audition", "interview", "defense", "thesis", "dissertation"
    ],
    "Original_contribution": [
        # Innovation Terms
        "original", "contribution", "innovation", "innovative", "novel", "pioneered", 
        "developed", "created", "invented", "discovery", "breakthrough", "patent", 
        "intellectual property", "IP", "method", "technique", "approach", "algorithm",
        "system", "framework", "architecture", "design", "solution", "implementation",
        
        # Impact Indicators
        "first", "pioneering", "groundbreaking", "revolutionary", "transformative", "disruptive",
        "game-changing", "paradigm-shifting", "cutting-edge", "leading-edge", "state-of-the-art",
        "next-generation", "advanced", "sophisticated", "complex", "unique", "distinctive",
        
        # Contribution Types
        "founded", "established", "launched", "initiated", "spearheaded", "led", "directed",
        "conceptualized", "formulated", "devised", "engineered", "architected", "designed",
        "developed", "implemented", "deployed", "commercialized", "scaled", "optimized"
    ],
    "Scholarly_articles": [
        # Publication Terms
        "publication", "published", "paper", "article", "journal", "conference", 
        "proceedings", "book", "chapter", "thesis", "dissertation", "research", 
        "author", "co-author", "citation", "cited", "h-index", "impact factor",
        "peer-reviewed", "scholarly", "academic", "scientific",
        
        # Publication Types
        "monograph", "textbook", "handbook", "manual", "guide", "review", "survey",
        "meta-analysis", "systematic review", "case study", "technical report", "white paper",
        "working paper", "preprint", "manuscript", "commentary", "editorial", "letter",
        
        # Publication Metrics
        "highly cited", "well-cited", "widely cited", "frequently cited", "influential",
        "seminal", "foundational", "groundbreaking", "landmark", "classic", "definitive",
        "authoritative", "comprehensive", "extensive", "in-depth", "rigorous"
    ],
    "Critical_employment": [
        # Position Terms
        "position", "role", "job", "employment", "employed", "work", "career", 
        "professional", "experience", "senior", "lead", "chief", "head", "director", 
        "manager", "executive", "officer", "principal", "founder", "co-founder",
        "leadership", "team lead", "supervisor", "administrator",
        
        # Responsibility Indicators
        "responsible for", "accountable for", "in charge of", "oversaw", "managed",
        "led", "directed", "supervised", "coordinated", "administered", "executed",
        "implemented", "delivered", "achieved", "accomplished", "spearheaded",
        
        # Organizational Context
        "department", "division", "unit", "team", "group", "organization", "company",
        "corporation", "firm", "agency", "institution", "establishment", "enterprise",
        "venture", "startup", "practice", "office", "bureau", "laboratory", "studio"
    ],
    "High_remuneration": [
        # Financial Terms
        "salary", "compensation", "remuneration", "income", "earning", "wage", 
        "pay", "stipend", "bonus", "equity", "stock", "option", "benefit", 
        "package", "contract", "funding", "grant", "budget", "revenue", "profit",
        "million", "thousand", "dollar", "$", "€", "£", "¥",
        
        # Financial Qualifiers
        "high", "substantial", "significant", "considerable", "competitive", "premium",
        "top-tier", "executive", "senior", "director-level", "C-level", "leadership",
        
        # Value/Impact Indicators
        "managed", "oversaw", "controlled", "responsible for", "accountable for",
        "generated", "produced", "delivered", "achieved", "exceeded", "surpassed",
        "increased", "grew", "expanded", "improved", "enhanced", "optimized"
    ]
}

# Domain-specific entities for each criterion
DOMAIN_ENTITIES = {
    "Awards": [
        # Academic/Research
        "ACM", "IEEE", "Nobel", "Turing", "Fields Medal", "MacArthur", "Guggenheim",
        "Forbes 30", "Under 30", "Best Paper", "Distinguished", "Excellence", "Outstanding",
        "NeurIPS", "ICML", "ACL", "CVPR", "ICLR", "AAAI", "IJCAI", "Spotlight",
        
        # Business/Corporate
        "Fortune", "Inc 500", "Ernst & Young", "Entrepreneur of the Year", "Stevie Award",
        "CEO of the Year", "CIO of the Year", "CFO of the Year", "CMO of the Year",
        "Business Leader", "Innovator", "Disruptor", "Visionary", "Thought Leader",
        
        # Arts/Entertainment
        "Grammy", "Emmy", "Oscar", "Tony", "Golden Globe", "Cannes", "Sundance",
        "Pulitzer", "Booker Prize", "National Book Award", "BAFTA", "MTV", "Billboard",
        
        # Medicine/Healthcare
        "AMA", "NIH", "Lasker Award", "Breakthrough Prize", "Gairdner", "Physician of the Year",
        "Healthcare Leader", "Medical Innovation", "Clinical Excellence",
        
        # Law
        "ABA", "Super Lawyers", "Best Lawyers", "Chambers", "Legal 500", "Trial Lawyer of the Year",
        
        # Finance
        "CFA", "Wall Street", "Institutional Investor", "Hedge Fund", "Private Equity", "Venture Capital",
        
        # Technology
        "Apple Design Award", "Microsoft MVP", "Google Developer Expert", "Red Dot", "iF Design Award",
        "Tech Innovator", "CTO of the Year", "Product of the Year",
        
        # Education
        "Teacher of the Year", "Professor of the Year", "Dean's Award", "Chancellor's Award",
        "Educational Leadership", "Curriculum Innovation",
        
        # Government/Public Service
        "Presidential Medal", "Congressional", "Diplomatic Service", "Public Service Award",
        "Humanitarian Award", "Community Service", "Civic Leadership"
    ],
    
    "Membership": [
        # Academic/Research
        "ACM", "IEEE", "AAAI", "SIGKDD", "SIGGRAPH", "SIGCHI", "SIGIR", "SIGCOMM",
        "Association for Computing Machinery", "American Association for Artificial Intelligence",
        "Fellow", "Senior Member", "Board Member", "Committee Member", "Editorial Board",
        
        # Business/Corporate
        "Chamber of Commerce", "Business Roundtable", "World Economic Forum", "Davos",
        "YPO", "Young Presidents Organization", "EO", "Entrepreneurs Organization",
        "Board of Directors", "Advisory Board", "Executive Committee", "Leadership Council",
        
        # Professional Associations
        "AMA", "American Medical Association", "ABA", "American Bar Association",
        "AICPA", "CPA", "American Institute of Certified Public Accountants",
        "SHRM", "Society for Human Resource Management", "PMI", "Project Management Institute",
        "ASME", "American Society of Mechanical Engineers", "AIA", "American Institute of Architects",
        
        # Industry Groups
        "Industry Association", "Trade Association", "Consortium", "Alliance", "Coalition",
        "Standards Body", "Working Group", "Special Interest Group",
        
        # Non-Profit/NGO
        "Non-Profit Board", "Foundation Board", "Trustee", "NGO", "Charity Board",
        "Volunteer Leadership", "Community Organization", "Social Impact"
    ],
    
    "Press": [
        # Major Publications
        "MIT Technology Review", "Wired", "TechCrunch", "Nature", "Science", "Forbes",
        "Wall Street Journal", "New York Times", "Washington Post", "Bloomberg", "Reuters",
        "Financial Times", "Economist", "Harvard Business Review", "Fast Company", "Inc Magazine",
        
        # Broadcast Media
        "CNN", "CNBC", "BBC", "NPR", "PBS", "60 Minutes", "Today Show", "Good Morning America",
        "Fox News", "ABC", "NBC", "CBS", "HBO", "Netflix", "Documentary",
        
        # Speaking Engagements
        "TED", "TEDx", "Keynote", "Invited Talk", "Panel", "Panelist", "Interview", "Featured",
        "Spotlight", "Innovator", "Pioneer", "Thought Leader", "Expert Commentary",
        
        # Industry Events
        "Conference", "Summit", "Forum", "Symposium", "Convention", "Expo", "Trade Show",
        "Webinar", "Podcast", "Roundtable", "Town Hall", "Fireside Chat",
        
        # Social Media
        "Influencer", "Viral", "Trending", "LinkedIn", "Twitter", "Instagram", "YouTube",
        "Followers", "Subscribers", "Channel", "Blog", "Vlog", "Content Creator"
    ],
    
    "Judging": [
        # Academic/Research
        "Program Committee", "Editorial Board", "Associate Editor", "Editor", "Reviewer",
        "NeurIPS", "ICML", "ACL", "CVPR", "ICLR", "AAAI", "IJCAI", "JMLR", "TPAMI",
        "Transactions", "Journal", "Conference", "Workshop", "Symposium",
        
        # Business/Corporate
        "Selection Committee", "Award Committee", "Grant Committee", "Investment Committee",
        "Venture Capital", "Angel Investor", "Startup Competition", "Pitch Competition",
        "Hackathon", "Innovation Challenge", "Business Plan Competition",
        
        # Arts/Entertainment
        "Festival Jury", "Film Festival", "Art Competition", "Design Competition", "Literary Prize",
        "Music Competition", "Talent Show", "Reality Competition", "Curator", "Artistic Director",
        
        # Professional
        "Hiring Committee", "Promotion Committee", "Tenure Committee", "Admissions Committee",
        "Certification Board", "Accreditation", "Professional Standards", "Ethics Committee",
        "Disciplinary Board", "Licensing Board", "Examination Board",
        
        # Community/Public
        "Scholarship Committee", "Fellowship Committee", "Community Grant", "Public Art",
        "Civic Award", "Volunteer Recognition", "Nonprofit Award"
    ],
    
    "Original_contribution": [
        # Technology/Engineering
        "Patent", "Algorithm", "Framework", "Architecture", "System", "Pipeline",
        "BERT", "Transformer", "CNN", "RNN", "GAN", "Reinforcement Learning", "Transfer Learning",
        "Few-shot", "Zero-shot", "Self-supervised", "Unsupervised", "Semi-supervised",
        "State-of-the-art", "SOTA", "Benchmark", "Accuracy", "Precision", "Recall", "F1",
        
        # Business/Corporate
        "Business Model", "Revenue Model", "Go-to-Market", "Strategy", "Disruption",
        "Market Creation", "Category Creation", "Industry Transformation", "Paradigm Shift",
        "Operational Excellence", "Process Innovation", "Supply Chain", "Customer Experience",
        
        # Product/Design
        "Product Innovation", "Design Innovation", "User Experience", "User Interface",
        "Industrial Design", "Service Design", "Experience Design", "Design Thinking",
        "Human-Centered Design", "Accessibility", "Inclusive Design", "Sustainable Design",
        
        # Creative/Artistic
        "Creative Direction", "Artistic Innovation", "Narrative Innovation", "Visual Language",
        "Stylistic Innovation", "Genre-Defining", "Medium-Defining", "Technique Innovation",
        
        # Scientific/Medical
        "Scientific Discovery", "Medical Breakthrough", "Clinical Innovation", "Therapeutic Approach",
        "Diagnostic Method", "Treatment Protocol", "Drug Development", "Medical Device",
        "Biomarker", "Genetic Marker", "Clinical Trial", "Research Methodology",
        
        # Social/Educational
        "Social Innovation", "Educational Innovation", "Pedagogical Approach", "Curriculum Design",
        "Learning Methodology", "Assessment Methodology", "Community Development",
        "Policy Innovation", "Governance Model", "Public Service Model"
    ],
    
    "Scholarly_articles": [
        # Academic Journals
        "Nature", "Science", "Cell", "JMLR", "TPAMI", "NeurIPS", "ICML", "ACL", "CVPR",
        "Citation", "h-index", "Impact Factor", "First Author", "Corresponding Author",
        "Proceedings", "Transactions", "Journal", "Conference", "Workshop", "Symposium",
        
        # Business/Professional Publications
        "Harvard Business Review", "MIT Sloan Management Review", "California Management Review",
        "Journal of Marketing", "Journal of Finance", "Strategic Management Journal",
        "White Paper", "Industry Report", "Market Analysis", "Trend Report", "Case Study",
        
        # Legal Publications
        "Law Review", "Legal Journal", "Bar Journal", "Legal Commentary", "Legal Analysis",
        "Case Commentary", "Regulatory Analysis", "Policy Brief", "Legal Brief",
        
        # Medical/Healthcare Publications
        "JAMA", "New England Journal of Medicine", "The Lancet", "BMJ", "Clinical Trial",
        "Meta-analysis", "Systematic Review", "Clinical Study", "Cohort Study", "Case Report",
        
        # Technical Documentation
        "Technical Report", "Technical Specification", "Standard", "Protocol", "Best Practice",
        "Implementation Guide", "Reference Architecture", "API Documentation",
        
        # Books and Chapters
        "Book", "Textbook", "Monograph", "Edited Volume", "Chapter", "Handbook",
        "Encyclopedia Entry", "Reference Work", "Anthology", "Compilation"
    ],
    
    "Critical_employment": [
        # Technology Companies
        "Google", "Microsoft", "Apple", "Amazon", "Facebook", "Meta", "OpenAI", "DeepMind",
        "Research Scientist", "Principal Scientist", "Distinguished Scientist", "Fellow",
        "Chief Scientist", "Director of Research", "VP of Research", "Head of AI", "Lead",
        
        # Executive Roles
        "CEO", "Chief Executive Officer", "COO", "Chief Operating Officer", "CFO", "Chief Financial Officer",
        "CTO", "Chief Technology Officer", "CIO", "Chief Information Officer", "CISO", "Chief Information Security Officer",
        "CMO", "Chief Marketing Officer", "CPO", "Chief Product Officer", "CDO", "Chief Data Officer",
        "President", "Executive Director", "Managing Director", "General Manager", "Partner",
        
        # Leadership Roles
        "Director", "Vice President", "Senior Vice President", "Executive Vice President",
        "Head of", "Leader", "Manager", "Senior Manager", "Principal", "Group Lead",
        "Department Chair", "Dean", "Provost", "Chancellor", "President", "Superintendent",
        
        # Specialized Roles
        "Surgeon", "Chief of Surgery", "Chief Medical Officer", "Lead Attorney", "Managing Partner",
        "Senior Counsel", "General Counsel", "Chief Architect", "Lead Designer", "Creative Director",
        "Principal Engineer", "Distinguished Engineer", "Fellow", "Senior Fellow",
        
        # Industry-Specific
        "Portfolio Manager", "Fund Manager", "Investment Banker", "Private Equity", "Venture Capital",
        "Executive Chef", "Artistic Director", "Music Director", "Film Director", "Producer",
        "Editor-in-Chief", "Bureau Chief", "Foreign Correspondent", "Senior Correspondent"
    ],
    
    "High_remuneration": [
        # Financial Terms
        "Million", "Billion", "Grant", "Funding", "Venture Capital", "VC", "Series A",
        "Series B", "IPO", "Acquisition", "Equity", "Stock Options", "RSU", "Bonus",
        
        # Compensation Indicators
        "Compensation", "Salary", "Remuneration", "Package", "Base Salary", "Total Compensation",
        "Annual Compensation", "Executive Compensation", "C-Suite Compensation",
        
        # Performance Indicators
        "Revenue Growth", "Profit Margin", "EBITDA", "Market Share", "Cost Reduction",
        "Efficiency Improvement", "Productivity Gain", "ROI", "Return on Investment",
        
        # Budget/Resource Control
        "Budget", "P&L", "Profit and Loss", "Revenue", "Portfolio", "Assets Under Management",
        "AUM", "Fund Size", "Capital", "Managed Budget", "Departmental Budget", "Project Budget",
        
        # Value Creation
        "Value Creation", "Shareholder Value", "Market Capitalization", "Enterprise Value",
        "Valuation", "Company Valuation", "Exit Value", "Acquisition Value", "Deal Size"
    ]
}

# Try to import spaCy for NER
try:
    import spacy
    SPACY_AVAILABLE = True
    # Try to load a spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded spaCy model for NER")
    except:
        logger.warning("Could not load spaCy model. Attempting to download it...")
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
            logger.info("Successfully downloaded and loaded spaCy model")
        except:
            logger.error("Failed to download spaCy model. NER will not be available.")
            SPACY_AVAILABLE = False
except ImportError:
    logger.warning("spaCy not installed. NER will not be available.")
    SPACY_AVAILABLE = False
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
        import spacy
        SPACY_AVAILABLE = True
        # Try to download a model
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")
            logger.info("Successfully installed spaCy and downloaded model")
        except:
            logger.error("Failed to download spaCy model. NER will not be available.")
            SPACY_AVAILABLE = False
    except:
        logger.error("Failed to install spaCy. NER will not be available.")
        SPACY_AVAILABLE = False

def parse_cv(file_path: str) -> Dict:
    """
    Parse a CV file and extract text content, categorizing sentences by O-1A criteria.
    Supports PDF, DOCX, and TXT formats.
    
    Args:
        file_path: Path to the CV file
        
    Returns:
        Dictionary containing the full CV text and categorized sentences
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        logger.info(f"Parsing CV file: {file_path} with extension {file_extension}")
        
        if file_extension == '.pdf':
            text = _parse_pdf(file_path)
        elif file_extension == '.docx':
            text = _parse_docx(file_path)
        elif file_extension == '.txt':
            text = _parse_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Preprocess the extracted text
        text = _preprocess_text(text)
        
        # Extract named entities if spaCy is available
        entities = {}
        if SPACY_AVAILABLE:
            entities = _extract_named_entities(text)
            logger.info(f"Extracted {sum(len(ents) for ents in entities.values())} named entities")
        
        # Categorize sentences by criteria
        categorized_sentences = _categorize_sentences(text, entities)
        
        logger.info(f"Successfully parsed CV with {len(text)} characters")
        
        return {
            "full_text": text,
            "categorized_sentences": categorized_sentences,
            "entities": entities
        }
    except Exception as e:
        logger.error(f"Error parsing CV file {file_path}: {str(e)}", exc_info=True)
        raise ValueError(f"Failed to parse CV file: {str(e)}")

def _extract_named_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities from the CV text using spaCy.
    
    Args:
        text: CV text
        
    Returns:
        Dictionary mapping entity types to lists of entities
    """
    if not SPACY_AVAILABLE:
        return {}
    
    # Process the text with spaCy
    doc = nlp(text)
    
    # Extract entities
    entities = {
        "ORG": [],  # Organizations
        "PERSON": [],  # People
        "GPE": [],  # Geopolitical entities (countries, cities)
        "LOC": [],  # Locations
        "PRODUCT": [],  # Products
        "EVENT": [],  # Events
        "WORK_OF_ART": [],  # Titles of books, songs, etc.
        "LAW": [],  # Laws, regulations
        "LANGUAGE": [],  # Languages
        "DATE": [],  # Dates
        "TIME": [],  # Times
        "PERCENT": [],  # Percentages
        "MONEY": [],  # Monetary values
        "QUANTITY": [],  # Quantities
        "ORDINAL": [],  # Ordinal numbers
        "CARDINAL": []  # Cardinal numbers
    }
    
    # Collect entities
    for ent in doc.ents:
        if ent.label_ in entities:
            # Normalize and deduplicate
            entity_text = ent.text.strip()
            if entity_text and entity_text not in entities[ent.label_]:
                entities[ent.label_].append(entity_text)
    
    return entities

def _parse_pdf(file_path: str) -> str:
    """Parse PDF files using PyPDF."""
    try:
        from pypdf import PdfReader
        
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        
        if not text.strip():
            logger.warning("PDF text extraction yielded empty result, trying alternative method")
            # Try alternative extraction method if available
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
            except ImportError:
                logger.warning("pdfplumber not available for alternative PDF extraction")
        
        return text
    except ImportError:
        logger.error("PyPDF not installed. Attempting to install...")
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pypdf"])
            from pypdf import PdfReader
            
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
            return text
        except Exception as e:
            logger.error(f"Failed to install or use PyPDF: {str(e)}")
            raise ImportError(f"PyPDF is required but not available: {str(e)}")

def _parse_docx(file_path: str) -> str:
    """Parse DOCX files using docx2txt."""
    try:
        import docx2txt
        
        text = docx2txt.process(file_path)
        return text
    except ImportError:
        logger.error("docx2txt not installed. Attempting to install...")
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "docx2txt"])
            import docx2txt
            
            text = docx2txt.process(file_path)
            return text
        except Exception as e:
            logger.error(f"Failed to install or use docx2txt: {str(e)}")
            raise ImportError(f"docx2txt is required but not available: {str(e)}")

def _parse_txt(file_path: str) -> str:
    """Parse TXT files by reading the content directly."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except UnicodeDecodeError:
        # Try different encodings if UTF-8 fails
        encodings = ['latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                logger.info(f"Successfully read text file with {encoding} encoding")
                return text
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, try binary mode and decode with errors='replace'
        with open(file_path, 'rb') as f:
            binary_data = f.read()
        text = binary_data.decode('utf-8', errors='replace')
        logger.warning("Used fallback binary reading with replacement for text file")
        return text

def _preprocess_text(text: str) -> str:
    """
    Preprocess the extracted text to improve quality.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Preprocessed text
    """
    if not text:
        return ""
    
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Fix common OCR/extraction issues
    text = text.replace('•', '- ')  # Replace bullets with dashes
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between lowercase and uppercase letters
    
    # Ensure section headers are properly separated
    section_headers = ['EDUCATION', 'EXPERIENCE', 'SKILLS', 'PUBLICATIONS', 'AWARDS', 'CERTIFICATIONS', 
                      'PROJECTS', 'LANGUAGES', 'REFERENCES', 'SUMMARY', 'OBJECTIVE', 'PROFESSIONAL EXPERIENCE',
                      'WORK EXPERIENCE', 'EMPLOYMENT', 'QUALIFICATIONS', 'ACHIEVEMENTS', 'HONORS']
    
    for header in section_headers:
        # Make sure section headers have newlines before and after
        text = re.sub(r'([^\n])(' + header + r')([^\n])', r'\1\n\2\n\3', text, flags=re.IGNORECASE)
    
    return text

def _categorize_sentences(text: str, entities: Dict[str, List[str]] = None) -> Dict[str, List[str]]:
    """
    Categorize sentences in the CV by O-1A criteria.
    
    Args:
        text: Preprocessed CV text
        entities: Named entities extracted from the text (optional)
        
    Returns:
        Dictionary mapping each criterion to a list of relevant sentences
    """
    # Initialize result dictionary
    categorized = {criterion: [] for criterion in O1A_CRITERIA}
    
    # Extract sections from the CV
    sections = _extract_sections(text)
    
    # Process each section
    for section_name, section_content in sections.items():
        # Determine which criteria this section might be relevant to
        relevant_criteria = _get_relevant_criteria_for_section(section_name)
        
        # Split section into sentences or bullet points
        items = _split_into_items(section_content)
        
        # Categorize each item
        for item in items:
            item = item.strip()
            if not item:
                continue
                
            # Check which criteria this item matches
            matched_criteria = _match_item_to_criteria(item, relevant_criteria, entities)
            
            # Add the item to each matched criterion
            for criterion in matched_criteria:
                categorized[criterion].append(item)
    
    # Log the results
    for criterion, items in categorized.items():
        logger.info(f"Found {len(items)} items for criterion: {criterion}")
    
    return categorized

def _extract_sections(text: str) -> Dict[str, str]:
    """
    Extract sections from the CV.
    
    Args:
        text: CV text
        
    Returns:
        Dictionary mapping section names to their content
    """
    # Common section names in CVs
    section_patterns = [
        (r'AWARDS|HONORS|RECOGNITIONS|PRIZES', "Awards"),
        (r'MEMBERSHIP|PROFESSIONAL MEMBERSHIPS|AFFILIATIONS|PROFESSIONAL AFFILIATIONS', "Memberships"),
        (r'PUBLICATIONS|SCHOLARLY ARTICLES|PAPERS|RESEARCH PAPERS|JOURNAL ARTICLES', "Publications"),
        (r'PRESS|MEDIA|MEDIA COVERAGE|INTERVIEWS|PUBLIC APPEARANCES|INVITED TALKS|PRESENTATIONS', "Press"),
        (r'JUDGING|REVIEWER|EDITORIAL|REVIEW ACTIVITIES', "Judging"),
        (r'PATENTS|INNOVATIONS|INVENTIONS|ORIGINAL CONTRIBUTIONS|RESEARCH', "Innovations"),
        (r'EMPLOYMENT|PROFESSIONAL EXPERIENCE|WORK EXPERIENCE|EXPERIENCE', "Employment"),
        (r'SALARY|COMPENSATION|REMUNERATION', "Compensation"),
        (r'EDUCATION|ACADEMIC BACKGROUND', "Education"),
        (r'SKILLS|TECHNICAL SKILLS|COMPETENCIES', "Skills"),
        (r'SUMMARY|PROFILE|OBJECTIVE', "Summary"),
        (r'PROJECTS|PROFESSIONAL PROJECTS', "Projects"),
        (r'CERTIFICATIONS|LICENSES', "Certifications"),
        (r'LANGUAGES|LANGUAGE SKILLS', "Languages"),
        (r'REFERENCES', "References")
    ]
    
    # Split the text into lines
    lines = text.split('\n')
    
    # Initialize result
    sections = {}
    current_section = "Uncategorized"
    current_content = []
    
    # Process each line
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Check if this line is a section header
        is_header = False
        for pattern, section_name in section_patterns:
            if re.search(pattern, line.upper()):
                # If we have content for the previous section, add it
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start a new section
                current_section = section_name
                current_content = []
                is_header = True
                break
        
        # If not a header, add to current section content
        if not is_header:
            current_content.append(line)
    
    # Add the last section
    if current_content:
        sections[current_section] = '\n'.join(current_content)
    
    return sections

def _get_relevant_criteria_for_section(section_name: str) -> List[str]:
    """
    Determine which criteria a section might be relevant to.
    
    Args:
        section_name: Name of the section
        
    Returns:
        List of potentially relevant criteria
    """
    section_to_criteria = {
        "Awards": ["Awards"],
        "Memberships": ["Membership"],
        "Publications": ["Scholarly_articles"],
        "Press": ["Press"],
        "Judging": ["Judging"],
        "Innovations": ["Original_contribution"],
        "Employment": ["Critical_employment", "High_remuneration"],
        "Compensation": ["High_remuneration"],
        "Education": ["Scholarly_articles"],
        "Summary": O1A_CRITERIA,  # Summary might mention any criterion
        "Projects": ["Original_contribution"],
        "Certifications": ["Membership"],
        "Languages": [],
        "References": [],
        "Skills": ["Original_contribution"],
        "Uncategorized": O1A_CRITERIA  # Check all criteria for uncategorized content
    }
    
    return section_to_criteria.get(section_name, O1A_CRITERIA)

def _split_into_items(text: str) -> List[str]:
    """
    Split section content into individual items (sentences or bullet points).
    
    Args:
        text: Section content
        
    Returns:
        List of items
    """
    # First, try to split by bullet points
    bullet_pattern = r'(?:^|\n)(?:[-•*]\s+)(.+?)(?=(?:\n[-•*]\s+|\n\n|\Z))'
    bullet_items = re.findall(bullet_pattern, text, re.DOTALL)
    
    if bullet_items:
        # Also include any text before the first bullet point
        first_bullet_pos = text.find('-')
        if first_bullet_pos > 0:
            intro_text = text[:first_bullet_pos].strip()
            if intro_text:
                # Split intro text into sentences
                intro_sentences = re.split(r'(?<=[.!?])\s+', intro_text)
                return intro_sentences + bullet_items
        return bullet_items
    
    # If no bullet points, split by sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s for s in sentences if s.strip()]

def _match_item_to_criteria(item: str, potential_criteria: List[str], entities: Dict[str, List[str]] = None) -> List[str]:
    """
    Match an item to relevant criteria based on keywords and named entities.
    
    Args:
        item: Text item to categorize
        potential_criteria: List of criteria to check against
        entities: Named entities extracted from the text (optional)
        
    Returns:
        List of matching criteria
    """
    item_lower = item.lower()
    matched_criteria = []
    
    # Check for domain-specific entities first
    for criterion in potential_criteria:
        # Check domain-specific entities
        domain_entities = DOMAIN_ENTITIES.get(criterion, [])
        entity_matches = sum(1 for entity in domain_entities if entity.lower() in item_lower)
        
        # If we have domain entity matches, this is a strong signal
        if entity_matches >= 1:
            matched_criteria.append(criterion)
            continue
        
        # Check if any keywords for this criterion appear in the item
        keywords = CRITERIA_KEYWORDS.get(criterion, [])
        
        # Count how many keywords match
        match_count = sum(1 for keyword in keywords if keyword.lower() in item_lower)
        
        # If we have a good number of matches, or a very specific match, include this criterion
        if match_count >= 2 or any(re.search(r'\b' + re.escape(keyword.lower()) + r'\b', item_lower) for keyword in keywords):
            matched_criteria.append(criterion)
    
    # If we have named entities, use them to enhance matching
    if entities and not matched_criteria:
        # Check if any organizations in the item match known important organizations
        for org in entities.get("ORG", []):
            org_lower = org.lower()
            # Check if this organization appears in the item
            if org_lower in item_lower:
                # Check which criteria this organization might be relevant to
                for criterion, domain_entities in DOMAIN_ENTITIES.items():
                    if criterion in potential_criteria:
                        for entity in domain_entities:
                            if entity.lower() in org_lower:
                                matched_criteria.append(criterion)
                                break
        
        # Check for monetary values (relevant to High_remuneration)
        if "High_remuneration" in potential_criteria and (entities.get("MONEY", []) or entities.get("CARDINAL", [])):
            for money in entities.get("MONEY", []) + entities.get("CARDINAL", []):
                if money.lower() in item_lower and any(char.isdigit() for char in money):
                    matched_criteria.append("High_remuneration")
                    break
        
        # Check for percentages (often relevant to Original_contribution)
        if "Original_contribution" in potential_criteria and entities.get("PERCENT", []):
            for percent in entities.get("PERCENT", []):
                if percent.lower() in item_lower:
                    matched_criteria.append("Original_contribution")
                    break
    
    # If no matches but the item is in a relevant section, use heuristics
    if not matched_criteria and len(potential_criteria) <= 2:
        # If we're in a specific section, default to that section's primary criterion
        matched_criteria = potential_criteria[:1]  # Take the first criterion as default
    
    return matched_criteria 