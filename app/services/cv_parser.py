import os
from typing import Optional

def parse_cv(file_path: str) -> str:
    """
    Parse a CV file and extract text content.
    Supports PDF, DOCX, and TXT formats.
    
    Args:
        file_path: Path to the CV file
        
    Returns:
        Extracted text from the CV
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return _parse_pdf(file_path)
    elif file_extension == '.docx':
        return _parse_docx(file_path)
    elif file_extension == '.txt':
        return _parse_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def _parse_pdf(file_path: str) -> str:
    """Parse PDF files using PyPDF."""
    from pypdf import PdfReader
    
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def _parse_docx(file_path: str) -> str:
    """Parse DOCX files using docx2txt."""
    import docx2txt
    
    text = docx2txt.process(file_path)
    return text

def _parse_txt(file_path: str) -> str:
    """Parse TXT files by reading the content directly."""
    with open(file_path, 'r') as f:
        text = f.read()
    return text 