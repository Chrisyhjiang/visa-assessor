import os
import re
import json
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_legal_documents(docs_dir: str) -> List[str]:
    """
    Load O-1A legal documents and guidelines
    """
    documents = []
    
    for filename in os.listdir(docs_dir):
        if filename.endswith('.txt') or filename.endswith('.md'):
            with open(os.path.join(docs_dir, filename), 'r', encoding='utf-8') as f:
                documents.append(f.read())
    
    return documents

def split_documents(documents: List[str]) -> List[Dict[str, Any]]:
    """
    Split documents into chunks for embedding
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for doc in documents:
        doc_chunks = text_splitter.split_text(doc)
        for chunk in doc_chunks:
            chunks.append({
                "text": chunk,
                "metadata": {}  # Can add source info if needed
            })
    
    return chunks

def process_and_save_chunks(docs_dir: str, output_file: str):
    """
    Process legal documents and save chunks for embedding
    """
    documents = load_legal_documents(docs_dir)
    chunks = split_documents(documents)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2)
    
    print(f"Processed {len(documents)} documents into {len(chunks)} chunks")

if __name__ == "__main__":
    process_and_save_chunks(
        docs_dir="data/legal_docs",
        output_file="data/rag/legal_chunks.json"
    ) 