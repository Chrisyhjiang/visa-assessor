"""
Text Extraction Utility

This module provides utilities for extracting text content from various document formats
(PDF, DOCX, TXT) that are commonly used for CVs and resumes. It handles the complexities
of different file formats and provides a unified interface for text extraction.

Supported formats:
- PDF (using PyPDF2)
- DOCX (using python-docx)
- TXT (direct UTF-8 decoding)
"""

import PyPDF2
import docx
from io import BytesIO
from typing import List

def extract_text_from_pdf(content: bytes) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        content (bytes): Raw bytes of the PDF file
        
    Returns:
        str: Extracted text content
        
    Raises:
        Exception: If text extraction fails
    """
    text = ""
    try:
        pdf_file = BytesIO(content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text() + "\n"
            
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")
        
    return text

def extract_text_from_docx(content: bytes) -> str:
    """
    Extract text content from a DOCX file.
    
    Args:
        content (bytes): Raw bytes of the DOCX file
        
    Returns:
        str: Extracted text content
        
    Raises:
        Exception: If text extraction fails
    """
    text = ""
    try:
        doc_file = BytesIO(content)
        doc = docx.Document(doc_file)
        
        for para in doc.paragraphs:
            text += para.text + "\n"
            
    except Exception as e:
        raise Exception(f"Error extracting text from DOCX: {str(e)}")
        
    return text

def extract_text(content: bytes, file_extension: str) -> str:
    """
    Extract text from a CV file based on its file type.
    
    This function serves as the main entry point for text extraction,
    delegating to specific extractors based on the file extension.
    
    Args:
        content (bytes): Raw bytes of the file
        file_extension (str): File extension (pdf, docx, or txt)
        
    Returns:
        str: Extracted text content
        
    Raises:
        ValueError: If file format is unsupported
        Exception: If text extraction fails
    """
    if file_extension == "pdf":
        return extract_text_from_pdf(content)
    elif file_extension == "docx":
        return extract_text_from_docx(content)
    elif file_extension == "txt":
        return content.decode("utf-8")
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for processing.
    
    This function is useful when processing large documents that need to be
    broken down into smaller pieces for analysis, while maintaining context
    through overlapping chunks.
    
    Args:
        text (str): Input text to be split
        chunk_size (int, optional): Maximum number of words per chunk. Defaults to 500.
        overlap (int, optional): Number of words to overlap between chunks. Defaults to 50.
        
    Returns:
        List[str]: List of text chunks
    """
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [text]
        
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
        
    return chunks 