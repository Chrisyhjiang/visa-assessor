"""
BGE-based retriever for RAG
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Try to import FAISS and sentence-transformers
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    print("Warning: FAISS or sentence-transformers not available. BGE retriever will not work.")

class BGERetriever:
    def __init__(self, vector_db_dir: str):
        """
        Initialize the retriever with a pre-built vector database
        """
        if not DEPENDENCIES_AVAILABLE:
            raise ImportError("FAISS or sentence-transformers not available. Please install them.")
        
        # Check if vector database exists
        index_path = os.path.join(vector_db_dir, "legal_docs.index")
        texts_path = os.path.join(vector_db_dir, "legal_docs_texts.pkl")
        model_path = os.path.join(vector_db_dir, "bge_model")
        
        if not os.path.exists(index_path) or not os.path.exists(texts_path):
            raise FileNotFoundError(f"Vector database files not found in {vector_db_dir}")
        
        # Load the index
        self.index = faiss.read_index(index_path)
        
        # Load the texts
        with open(texts_path, 'rb') as f:
            self.texts = pickle.load(f)
        
        # Load the embedding model if it exists, otherwise download it
        if os.path.exists(model_path):
            print(f"Loading BGE model from {model_path}")
            self.model = SentenceTransformer(model_path)
        else:
            print("BGE model not found locally, downloading from HuggingFace")
            self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        
        print(f"BGE retriever initialized with {len(self.texts)} documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieve relevant legal document chunks for a query using BGE embeddings
        """
        # Generate query embedding
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search the index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Get the relevant texts
        relevant_texts = [self.texts[idx] for idx in indices[0]]
        
        return relevant_texts

def get_relevant_legal_context(retriever: BGERetriever, cv_text: str) -> str:
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