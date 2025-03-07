"""
Test script for the O-1A visa assessment API.
This script tests the API without requiring a GPU by mocking the LLM.
"""

import os
import json
import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
import tempfile
from typing import Dict, Any
import logging

# Import our processing modules
from processors.document_parser import extract_text_from_document
from processors.prompt_builder import build_o1a_assessment_prompt, build_o1a_assessment_prompt_with_rag
from processors.response_formatter import format_llm_response

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="O-1A Visa Qualification Assessment API (Test Mode)",
    description="API for assessing O-1A visa qualification based on CV analysis (Test Mode)",
    version="1.0.0"
)

# Mock LLM response
MOCK_LLM_RESPONSE = """
{
  "criteria_matches": {
    "awards": [
      {
        "text": "ACM Distinguished Researcher Award, 2022",
        "explanation": "This is a nationally recognized award in the field of computer science, demonstrating excellence and recognition by peers."
      },
      {
        "text": "Forbes 30 Under 30 in Science, 2018",
        "explanation": "This is a prestigious recognition by a major media outlet for outstanding achievements in science."
      }
    ],
    "membership": [
      {
        "text": "Senior Member, Association for Computing Machinery (ACM)",
        "explanation": "ACM Senior Membership requires demonstrated excellence and is reviewed by peers in the field."
      }
    ],
    "press": [
      {
        "text": "Featured in MIT Technology Review, '35 Innovators Under 35,' 2020",
        "explanation": "MIT Technology Review is a major technology publication, and this feature specifically highlights the applicant's work and achievements."
      }
    ],
    "judging": [
      {
        "text": "Program Committee Member, NeurIPS (2018-present)",
        "explanation": "As a program committee member, the applicant evaluates and judges the work of others in the field for acceptance to a prestigious conference."
      }
    ],
    "original_contribution": [
      {
        "text": "Pioneered a novel approach to few-shot learning that reduced training data requirements by 60%",
        "explanation": "This represents an original contribution with quantifiable impact in the field of machine learning."
      }
    ],
    "scholarly_articles": [
      {
        "text": "Published 23 total papers, including in Nature Machine Intelligence, ACL, NeurIPS, and ICML",
        "explanation": "These are top-tier journals and conferences in the field, demonstrating scholarly contributions at the highest level."
      }
    ],
    "critical_employment": [
      {
        "text": "Lead a team of 12 researchers and engineers developing state-of-the-art NLP models at TechVision AI",
        "explanation": "This leadership role demonstrates employment in a critical capacity, directing significant research efforts."
      }
    ],
    "high_remuneration": [
      {
        "text": "Current annual compensation package at TechVision AI: $385,000",
        "explanation": "This compensation is significantly higher than the average for data scientists and AI researchers, indicating recognition of exceptional value."
      }
    ]
  },
  "qualification_rating": "High",
  "rating_justification": "The applicant meets all eight criteria for O-1A visa qualification with strong evidence in each category. The applicant has received prestigious awards, holds memberships in distinguished organizations, has been featured in major media, has judged the work of others, has made original contributions to the field, has published scholarly articles, has held critical employment positions, and commands a high salary.",
  "criteria_met_count": 8
}
"""

# Global variables for RAG
use_rag = False
retriever = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system at startup if enabled"""
    global retriever, use_rag
    
    # Check if we should use RAG
    use_rag = os.environ.get("USE_RAG", "false").lower() == "true"
    
    # Initialize RAG if enabled
    if use_rag:
        try:
            from rag.retriever import LegalDocRetriever
            logger.info("Initializing RAG retriever...")
            retriever = LegalDocRetriever(vector_db_dir="data/rag/vector_db")
            logger.info("RAG retriever initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG: {str(e)}")
            use_rag = False
            logger.warning("RAG disabled due to initialization error")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    global use_rag
    
    return {
        "message": "O-1A Visa Qualification Assessment API (Test Mode)",
        "version": "1.0.0",
        "status": "active",
        "config": {
            "rag_enabled": use_rag,
            "mock_llm": True
        },
        "endpoints": {
            "/assess": "POST - Assess O-1A qualification from CV"
        },
        "note": "This is a test version that uses a mock LLM response"
    }

@app.post("/assess")
async def assess_qualification(
    cv_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Assess O-1A visa qualification based on uploaded CV
    """
    global retriever, use_rag
    
    # Create temp file to store the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(cv_file.filename)[1]) as temp_file:
        content = await cv_file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        # Extract text from document
        cv_text = extract_text_from_document(temp_file_path)
        logger.info(f"Extracted {len(cv_text)} characters from CV")
        
        # Build prompt based on RAG availability
        if use_rag and retriever is not None:
            from rag.retriever import get_relevant_legal_context
            legal_context = get_relevant_legal_context(retriever, cv_text)
            prompt = build_o1a_assessment_prompt_with_rag(cv_text, legal_context)
            logger.info("Using RAG-enhanced prompt")
        else:
            prompt = build_o1a_assessment_prompt(cv_text)
            logger.info("Using standard prompt")
        
        # In test mode, we use a mock LLM response instead of a real LLM
        logger.info("Using mock LLM response")
        llm_response = MOCK_LLM_RESPONSE
        
        # Clean and format the response
        formatted_response = format_llm_response(llm_response)
        
        # Schedule cleanup
        if background_tasks:
            background_tasks.add_task(os.unlink, temp_file_path)
        else:
            os.unlink(temp_file_path)
        
        return JSONResponse(content=formatted_response)
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        # Clean up the temp file in case of error
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred: {str(e)}"}
        )

if __name__ == "__main__":
    # Run the API
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting API on port {port}")
    logger.info(f"RAG enabled: {use_rag}")
    uvicorn.run(app, host="0.0.0.0", port=port) 