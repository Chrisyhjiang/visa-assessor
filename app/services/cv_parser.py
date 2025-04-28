import os
import logging
from typing import Optional, Dict

# Set up logging
logger = logging.getLogger(__name__)

def parse_cv(file_path: str) -> str:
    """
    Parse a CV file and extract its text content.
    
    Args:
        file_path: Path to the CV file (PDF, DOCX, or TXT)
        
    Returns:
        Text content of the CV
    """
    logger.info(f"Parsing CV file: {file_path}")
    
    # Get file extension
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Parse according to file type
    if file_extension == '.pdf':
        text = _parse_pdf(file_path)
    elif file_extension == '.docx':
        text = _parse_docx(file_path)
    elif file_extension == '.txt':
        text = _parse_txt(file_path)
    else:
        logger.error(f"Unsupported file format: {file_extension}")
        raise ValueError(f"Unsupported file format: {file_extension}. Please use PDF, DOCX, or TXT.")
    
    # Perform basic preprocessing
    text = _basic_text_cleanup(text)
    
    logger.info(f"Successfully parsed CV with {len(text)} characters")
    return text

def _parse_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        # Import PdfReader only when needed
        from pypdf import PdfReader
        logger.info("Parsing PDF document")
        
        reader = PdfReader(file_path)
        text_parts = []
        
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        
        full_text = "\n".join(text_parts)
        logger.info(f"Extracted {len(full_text)} characters from PDF")
        return full_text
    
    except Exception as e:
        logger.error(f"Error parsing PDF: {str(e)}")
        raise ValueError(f"Failed to parse PDF: {str(e)}")

def _parse_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    try:
        # Import docx2txt only when needed
        import docx2txt
        logger.info("Parsing DOCX document")
        
        text = docx2txt.process(file_path)
        logger.info(f"Extracted {len(text)} characters from DOCX")
        return text
    
    except Exception as e:
        logger.error(f"Error parsing DOCX: {str(e)}")
        raise ValueError(f"Failed to parse DOCX: {str(e)}")

def _parse_txt(file_path: str) -> str:
    """Read a plain text file."""
    try:
        logger.info("Reading TXT file")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        logger.info(f"Read {len(text)} characters from TXT file")
        return text
    
    except UnicodeDecodeError:
        # Try different encoding if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
            logger.info(f"Read {len(text)} characters from TXT file using latin-1 encoding")
            return text
        except Exception as e:
            logger.error(f"Error reading TXT file with latin-1 encoding: {str(e)}")
            raise ValueError(f"Failed to read TXT file: {str(e)}")
    except Exception as e:
        logger.error(f"Error reading TXT file: {str(e)}")
        raise ValueError(f"Failed to read TXT file: {str(e)}")

def _basic_text_cleanup(text: str) -> str:
    """Perform basic text cleaning operations."""
    if not text:
        return ""
    
    # Replace multiple spaces with a single space
    text = ' '.join(text.split())
    
    # Replace multiple newlines with a single newline
    import re
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove non-printable characters
    text = ''.join(c for c in text if c.isprintable() or c in ['\n', '\t'])
    
    return text 