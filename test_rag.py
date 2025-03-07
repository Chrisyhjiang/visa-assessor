#!/usr/bin/env python3
"""
Test script for the RAG retriever
"""

import os
import sys
from rag.retriever import LegalDocRetriever, get_relevant_legal_context

def main():
    print("Testing RAG retriever...")
    
    # Check if vector database exists
    vector_db_dir = "data/rag/vector_db"
    if not os.path.exists(vector_db_dir):
        print(f"Error: Vector database directory not found: {vector_db_dir}")
        return
    
    # Check for required files
    index_file = os.path.join(vector_db_dir, "legal_docs.index")
    texts_file = os.path.join(vector_db_dir, "legal_docs_texts.pkl")
    model_dir = os.path.join(vector_db_dir, "bge_model")
    
    if not os.path.exists(index_file):
        print(f"Error: Index file not found: {index_file}")
        return
    
    if not os.path.exists(texts_file):
        print(f"Error: Texts file not found: {texts_file}")
        return
    
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        return
    
    try:
        # Initialize the retriever
        print("Initializing retriever...")
        retriever = LegalDocRetriever(vector_db_dir)
        
        # Test retrieval
        print("\nTesting retrieval with a sample query...")
        query = "O-1A visa qualification criteria"
        results = retriever.retrieve(query, top_k=2)
        
        print(f"\nRetrieved {len(results)} results for query: '{query}'")
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(result[:200] + "..." if len(result) > 200 else result)
        
        # Test get_relevant_legal_context
        print("\nTesting get_relevant_legal_context...")
        cv_text = "Sample CV text for testing"
        legal_context = get_relevant_legal_context(retriever, cv_text)
        
        print(f"\nGenerated legal context ({len(legal_context)} characters)")
        print(legal_context[:500] + "..." if len(legal_context) > 500 else legal_context)
        
        print("\nRAG retriever test completed successfully!")
        
    except Exception as e:
        print(f"Error testing RAG retriever: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 