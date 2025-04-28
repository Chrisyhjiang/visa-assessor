from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings
from typing import List, Dict, Any, Optional
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

# Define constants
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
KNOWLEDGE_BASE_DIR = "app/knowledge_base"

class RAGService:
    def __init__(self, knowledge_base_dir: str = KNOWLEDGE_BASE_DIR):
        """
        Initialize the RAG service with BGE embeddings.
        
        Args:
            knowledge_base_dir: Directory containing knowledge base documents
        """
        try:
            if not sentence_transformers_available:
                raise ImportError("sentence_transformers package is required but not available")
            
            # Initialize text_splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                length_function=len,
            )
            
            # Initialize embeddings model
            self.embeddings = self._initialize_embeddings()
            
            # Initialize vector store
            self.vector_store = self._initialize_vector_store(knowledge_base_dir)
            
            # Cache for query results
            self.query_cache = {}
            
            logger.info("RAG service initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG service: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize RAG service: {str(e)}")
    
    def _initialize_embeddings(self) -> HuggingFaceBgeEmbeddings:
        """Initialize the embedding model with fallback options."""
        model_options = [
            "BAAI/bge-large-en-v1.5",
            "BAAI/bge-base-en-v1.5",
            "BAAI/bge-small-en-v1.5",
            "BAAI/bge-small-en"
        ]
        
        for model_name in model_options:
            try:
                logger.info(f"Trying to load embedding model: {model_name}")
                return HuggingFaceBgeEmbeddings(model_name=model_name)
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {str(e)}")
        
        # If all models fail, use the default model
        logger.warning(f"All models failed to load. Using default model: {DEFAULT_MODEL}")
        return HuggingFaceBgeEmbeddings(model_name=DEFAULT_MODEL)
    
    def _initialize_vector_store(self, knowledge_base_dir: str) -> FAISS:
        """Initialize the vector store with knowledge base documents."""
        documents = self._load_knowledge_base_documents(knowledge_base_dir)
        
        if not documents:
            logger.warning(f"No knowledge base documents found in {knowledge_base_dir}")
            documents = [Document(
                page_content="No knowledge base documents found.", 
                metadata={"criterion": "Unknown"}
            )]
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Create vector store
        return FAISS.from_documents(chunks, self.embeddings)
    
    def _load_knowledge_base_documents(self, knowledge_base_dir: str) -> List[Document]:
        """Load documents from the knowledge base directory."""
        documents = []
        
        try:
            for filename in os.listdir(knowledge_base_dir):
                if filename.endswith(".txt"):
                    file_path = os.path.join(knowledge_base_dir, filename)
                    try:
                        with open(file_path, "r") as f:
                            content = f.read()
                            # Extract criterion name from filename
                            criterion = os.path.splitext(filename)[0].capitalize()
                            
                            # Add the criterion name to the content
                            enhanced_content = f"O-1A Visa Criterion: {criterion}\n\n{content}"
                            
                            doc = Document(
                                page_content=enhanced_content,
                                metadata={"criterion": criterion, "source": file_path}
                            )
                            documents.append(doc)
                            logger.info(f"Loaded knowledge base document: {filename}")
                    except Exception as e:
                        logger.error(f"Error loading document {filename}: {str(e)}")
        except Exception as e:
            logger.error(f"Error accessing knowledge base directory: {str(e)}")
        
        return documents
    
    def query_knowledge_base(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the knowledge base for relevant information.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of relevant documents with metadata
        """
        # Check cache first
        cache_key = f"{query}_{top_k}"
        if cache_key in self.query_cache:
            logger.info(f"Using cached results for query: {query}")
            return self.query_cache[cache_key]
        
        # Enhance the query with O-1A context
        enhanced_query = f"O-1A visa qualification: {query}"
        logger.info(f"Querying knowledge base with: {enhanced_query}")
        
        # Perform the search
        results = self.vector_store.similarity_search_with_score(enhanced_query, k=top_k)
        
        # Process results
        processed_results = [
            {
                "content": doc.page_content,
                "criterion": doc.metadata.get("criterion", "Unknown"),
                "source": doc.metadata.get("source", "Unknown"),
                "score": score
            }
            for doc, score in results
        ]
        
        # Cache the results
        self.query_cache[cache_key] = processed_results
        
        return processed_results
    
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
            "Awards", "Membership", "Press", "Judging", 
            "Original_contribution", "Scholarly_articles", 
            "Critical_employment", "High_remuneration"
        ]
        
        # Split CV into chunks for processing
        cv_chunks = self.text_splitter.split_text(cv_text)
        logger.info(f"Split CV into {len(cv_chunks)} chunks for processing")
        
        results = {}
        
        # For each criterion, find relevant information in the CV
        for criterion in criteria:
            criterion_results = []
            
            # Query the knowledge base for information about this criterion
            kb_results = self.query_knowledge_base(
                f"Detailed explanation of {criterion} criterion for O-1A visa with examples", 
                top_k=2
            )
            
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