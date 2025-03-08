from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List, Dict, Any
import os
import logging
import importlib.util
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if sentence_transformers is available
sentence_transformers_available = importlib.util.find_spec("sentence_transformers") is not None
if not sentence_transformers_available:
    logger.warning("sentence_transformers package not found. Attempting to install it...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence_transformers"])
        logger.info("Successfully installed sentence_transformers")
        sentence_transformers_available = True
    except Exception as e:
        logger.error(f"Failed to install sentence_transformers: {str(e)}")

# Import HuggingFaceBgeEmbeddings only if sentence_transformers is available
if sentence_transformers_available:
    try:
        from langchain.embeddings import HuggingFaceBgeEmbeddings
        logger.info("Successfully imported HuggingFaceBgeEmbeddings")
    except ImportError as e:
        logger.error(f"Error importing HuggingFaceBgeEmbeddings: {str(e)}")
        raise ImportError(f"Error importing HuggingFaceBgeEmbeddings: {str(e)}")

class RAGService:
    def __init__(self, knowledge_base_dir: str = "app/knowledge_base"):
        """
        Initialize the RAG service with BGE embeddings.
        
        Args:
            knowledge_base_dir: Directory containing knowledge base documents
        """
        try:
            if not sentence_transformers_available:
                raise ImportError("sentence_transformers package is required but not available")
            
            # Initialize text_splitter before using it
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            
            # Initialize embeddings
            self.embeddings = HuggingFaceBgeEmbeddings(
                model_name="BAAI/bge-small-en"
            )
            
            # Initialize vector store
            self.vector_store = self._initialize_vector_store(knowledge_base_dir)
            
            logger.info("RAG service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG service: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize RAG service: {str(e)}")
    
    def _initialize_vector_store(self, knowledge_base_dir: str) -> FAISS:
        """
        Initialize the vector store with knowledge base documents.
        
        Args:
            knowledge_base_dir: Directory containing knowledge base documents
            
        Returns:
            Initialized FAISS vector store
        """
        documents = []
        
        # Load knowledge base documents
        for filename in os.listdir(knowledge_base_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(knowledge_base_dir, filename)
                with open(file_path, "r") as f:
                    content = f.read()
                    # Extract criterion name from filename (e.g., "awards.txt" -> "Awards")
                    criterion = os.path.splitext(filename)[0].capitalize()
                    doc = Document(
                        page_content=content,
                        metadata={"criterion": criterion}
                    )
                    documents.append(doc)
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Create vector store
        return FAISS.from_documents(chunks, self.embeddings)
    
    def query_knowledge_base(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the knowledge base for relevant information.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of relevant documents with metadata
        """
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        
        return [
            {
                "content": doc.page_content,
                "criterion": doc.metadata.get("criterion", "Unknown"),
                "score": score
            }
            for doc, score in results
        ]
    
    def process_cv(self, cv_text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process a CV and retrieve relevant information for each criterion.
        
        Args:
            cv_text: Text content of the CV
            
        Returns:
            Dictionary mapping criteria to relevant information
        """
        # Define the O-1A criteria
        criteria = [
            "Awards", 
            "Membership", 
            "Press", 
            "Judging", 
            "Original_contribution",
            "Scholarly_articles", 
            "Critical_employment", 
            "High_remuneration"
        ]
        
        # Split CV into chunks for processing
        cv_chunks = self.text_splitter.split_text(cv_text)
        
        results = {}
        
        # For each criterion, find relevant information in the CV
        for criterion in criteria:
            criterion_results = []
            
            # Query the knowledge base for information about this criterion
            kb_results = self.query_knowledge_base(f"What qualifies as {criterion} for O-1A visa?", top_k=2)
            
            # Use the knowledge base information to find relevant parts in the CV
            for kb_item in kb_results:
                for chunk in cv_chunks:
                    # Query combining the CV chunk and the criterion information
                    query = f"Does this CV section demonstrate {criterion} according to O-1A criteria? CV section: {chunk}"
                    
                    # We'll use this query with the LLM in the assessment service
                    criterion_results.append({
                        "cv_chunk": chunk,
                        "criterion_info": kb_item["content"],
                        "query": query
                    })
            
            results[criterion] = criterion_results
        
        return results 