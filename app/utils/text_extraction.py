import PyPDF2
import docx
from io import BytesIO
from typing import List

def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF file"""
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
    """Extract text from DOCX file"""
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
    """Extract text from CV file based on file type"""
    if file_extension == "pdf":
        return extract_text_from_pdf(content)
    elif file_extension == "docx":
        return extract_text_from_docx(content)
    elif file_extension == "txt":
        return content.decode("utf-8")
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for processing"""
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