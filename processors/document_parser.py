import PyPDF2
import docx
import os
from concurrent.futures import ThreadPoolExecutor

def extract_text_from_document(file_path: str) -> str:
    """
    Extract text from PDF, DOCX, or TXT files
    """ 
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension == '.txt':
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF file"""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        # Use multithreading for processing pages in parallel
        def extract_page_text(page_num):
            return reader.pages[page_num].extract_text()
        
        with ThreadPoolExecutor() as executor:
            text_chunks = list(executor.map(extract_page_text, range(len(reader.pages))))
            return "".join(text_chunks)

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX file"""
    doc = docx.Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read() 