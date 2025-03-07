"""
Simplified RAG retriever that doesn't rely on sentence-transformers
"""

import os
import json
import pickle
from typing import List, Dict, Any

class SimpleRetriever:
    def __init__(self, chunks_file: str):
        """
        Initialize the retriever with a JSON file containing chunks
        """
        # Load the chunks
        with open(chunks_file, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        
        # Extract the texts
        self.texts = [chunk["text"] for chunk in self.chunks]
        
        print(f"Loaded {len(self.texts)} chunks from {chunks_file}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Simple keyword-based retrieval
        """
        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Split query into keywords
        keywords = query_lower.split()
        
        # Score chunks based on keyword matches
        scores = []
        for text in self.texts:
            text_lower = text.lower()
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores.append(score)
        
        # Get indices of top_k highest scores
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # Get the relevant texts
        relevant_texts = [self.texts[idx] for idx in indices]
        
        return relevant_texts

def get_relevant_legal_context(retriever: SimpleRetriever, cv_text: str) -> str:
    """
    Get relevant legal context for a CV assessment
    """
    # Create queries for different aspects of O-1A criteria
    queries = [
        "O-1A visa qualification criteria",
        "O-1A extraordinary ability evidence requirements",
        "O-1A awards criterion",
        "O-1A membership criterion",
        "O-1A press coverage criterion",
        "O-1A judging criterion",
        "O-1A original contributions criterion",
        "O-1A scholarly articles criterion",
        "O-1A critical employment criterion",
        "O-1A high remuneration criterion"
    ]
    
    # Retrieve relevant chunks for each query
    all_relevant_chunks = []
    for query in queries:
        chunks = retriever.retrieve(query, top_k=2)  # Get top 2 for each query
        all_relevant_chunks.extend(chunks)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_chunks = []
    for chunk in all_relevant_chunks:
        if chunk not in seen:
            seen.add(chunk)
            unique_chunks.append(chunk)
    
    # Combine into a single context string
    legal_context = "\n\n".join(unique_chunks)
    
    return legal_context 