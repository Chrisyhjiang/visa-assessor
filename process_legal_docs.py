"""
Process legal documents for RAG.
This script processes the legal documents in data/legal_docs and creates the vector database.
"""

import os
import sys

def main():
    """Process legal documents and create vector database"""
    print("=== Processing Legal Documents for RAG ===")
    
    # Check if legal documents exist
    legal_docs_dir = "data/legal_docs"
    if not os.path.exists(legal_docs_dir):
        print(f"Error: Legal documents directory not found at {legal_docs_dir}")
        return False
    
    # Check if there are any documents
    doc_files = [f for f in os.listdir(legal_docs_dir) if f.endswith('.txt') or f.endswith('.md')]
    if not doc_files:
        print(f"Error: No text documents found in {legal_docs_dir}")
        return False
    
    print(f"Found {len(doc_files)} legal document(s): {', '.join(doc_files)}")
    
    # Create output directory
    os.makedirs("data/rag", exist_ok=True)
    
    # Process documents
    print("\n1. Processing documents into chunks...")
    try:
        from rag.document_processor import process_and_save_chunks
        process_and_save_chunks(
            docs_dir=legal_docs_dir,
            output_file="data/rag/legal_chunks.json"
        )
        print("✓ Successfully processed documents into chunks")
    except ImportError as e:
        print(f"Error: Required dependencies not installed: {e}")
        print("Please install the required dependencies with: pip install langchain")
        return False
    except Exception as e:
        print(f"Error processing documents: {e}")
        return False
    
    # Create vector database
    print("\n2. Creating vector database...")
    try:
        from rag.create_vector_db import create_vector_database
        create_vector_database(
            chunks_file="data/rag/legal_chunks.json",
            output_dir="data/rag/vector_db"
        )
        print("✓ Successfully created vector database")
    except ImportError as e:
        print(f"Error: Required dependencies not installed: {e}")
        print("Please install the required dependencies with: pip install sentence-transformers faiss-cpu")
        return False
    except Exception as e:
        print(f"Error creating vector database: {e}")
        return False
    
    print("\n=== Processing Complete ===")
    print("The RAG system is now ready to use.")
    print("You can run the API with RAG enabled using: USE_RAG=true python app.py")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 