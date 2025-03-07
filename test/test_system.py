"""
Test script for O-1A visa assessment system.
This script tests the core functionality without requiring a GPU.
"""

import os
import json
from processors.document_parser import extract_text_from_document
from processors.prompt_builder import build_o1a_assessment_prompt
from processors.response_formatter import format_llm_response

def test_document_parser():
    """Test the document parser with a sample CV"""
    sample_cv_path = "data/cv_samples/sample_cv.txt"
    if os.path.exists(sample_cv_path):
        cv_text = extract_text_from_document(sample_cv_path)
        print(f"Successfully extracted {len(cv_text)} characters from CV")
        print(f"First 100 characters: {cv_text[:100]}...")
        return True
    else:
        print(f"Error: Sample CV not found at {sample_cv_path}")
        return False

def test_prompt_builder():
    """Test the prompt builder with a sample CV text"""
    sample_cv_path = "data/cv_samples/sample_cv.txt"
    if os.path.exists(sample_cv_path):
        with open(sample_cv_path, 'r', encoding='utf-8') as f:
            cv_text = f.read()
        
        prompt = build_o1a_assessment_prompt(cv_text)
        print(f"Successfully built prompt with {len(prompt)} characters")
        print(f"First 100 characters of prompt: {prompt[:100]}...")
        return True
    else:
        print(f"Error: Sample CV not found at {sample_cv_path}")
        return False

def test_response_formatter():
    """Test the response formatter with a sample response"""
    sample_response = """
    {
      "criteria_matches": {
        "awards": [
          {
            "text": "ACM Distinguished Researcher Award, 2022",
            "explanation": "This is a nationally recognized award in the field of computer science."
          }
        ],
        "membership": [],
        "press": [],
        "judging": [],
        "original_contribution": [],
        "scholarly_articles": [],
        "critical_employment": [],
        "high_remuneration": []
      },
      "qualification_rating": "Low",
      "rating_justification": "The applicant only meets one criterion."
    }
    """
    
    formatted_response = format_llm_response(sample_response)
    print("Successfully formatted response:")
    print(json.dumps(formatted_response, indent=2))
    return True

def run_all_tests():
    """Run all tests"""
    print("=== Testing O-1A Visa Assessment System ===")
    
    print("\n1. Testing Document Parser...")
    parser_result = test_document_parser()
    
    print("\n2. Testing Prompt Builder...")
    prompt_result = test_prompt_builder()
    
    print("\n3. Testing Response Formatter...")
    formatter_result = test_response_formatter()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Document Parser: {'PASS' if parser_result else 'FAIL'}")
    print(f"Prompt Builder: {'PASS' if prompt_result else 'FAIL'}")
    print(f"Response Formatter: {'PASS' if formatter_result else 'FAIL'}")
    
    if parser_result and prompt_result and formatter_result:
        print("\nAll tests passed! The core system is working correctly.")
        print("To run the full system with LLM inference, you'll need a GPU.")
        print("You can rent a GPU on Lambda Labs or similar cloud providers.")
    else:
        print("\nSome tests failed. Please check the errors above.")

if __name__ == "__main__":
    run_all_tests() 