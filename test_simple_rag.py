#!/usr/bin/env python3
"""
Test script for the simplified RAG retriever
"""

import os
import sys
from rag.simple_retriever import SimpleRetriever, get_relevant_legal_context

def main():
    print("Testing simplified RAG retriever...")
    
    # Check if chunks file exists
    chunks_file = "data/rag/legal_chunks.json"
    if not os.path.exists(chunks_file):
        print(f"Error: Chunks file not found: {chunks_file}")
        return
    
    try:
        # Initialize the retriever
        print("Initializing retriever...")
        retriever = SimpleRetriever(chunks_file)
        
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
        
        print("\nSimplified RAG retriever test completed successfully!")
        
    except Exception as e:
        print(f"Error testing simplified RAG retriever: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 