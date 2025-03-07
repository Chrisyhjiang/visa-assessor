import json
from typing import List, Dict, Any
import numpy as np
import pickle
import os

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    raise ImportError(
        "Required dependencies not installed. "
        "Please install with: pip install sentence-transformers faiss-cpu"
    )

def create_vector_database(chunks_file: str, output_dir: str):
    """
    Create a FAISS vector database from document chunks using BGE embeddings
    """
    # Load document chunks
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Load BGE embedding model
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    
    # Extract texts for embedding
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity with normalized vectors
    faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
    index.add(embeddings)
    
    # Save index
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "legal_docs.index"))
    
    # Save texts and metadata for retrieval
    with open(os.path.join(output_dir, "legal_docs_texts.pkl"), 'wb') as f:
        pickle.dump(texts, f)
    
    # Save model for consistency
    model.save(os.path.join(output_dir, "bge_model"))
    
    print(f"Vector database created with {len(texts)} chunks")

if __name__ == "__main__":
    create_vector_database(
        chunks_file="data/rag/legal_chunks.json",
        output_dir="data/rag/vector_db"
    ) 