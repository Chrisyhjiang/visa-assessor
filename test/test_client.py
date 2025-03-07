"""
Test client for the O-1A visa assessment API.
This script tests the API by sending a sample CV.
"""

import requests
import json
import sys
import os

def test_api(api_url: str, cv_file_path: str):
    """Test the API by sending a sample CV"""
    print(f"Testing API at {api_url} with CV file {cv_file_path}")
    
    # Check if the CV file exists
    if not os.path.exists(cv_file_path):
        print(f"Error: CV file not found at {cv_file_path}")
        return False
    
    # Check if the API is running
    try:
        response = requests.get(api_url)
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            return False
        print("API is running")
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {api_url}")
        return False
    
    # Send the CV file to the API
    print("\nSending CV file to API...")
    files = {'cv_file': open(cv_file_path, 'rb')}
    try:
        response = requests.post(f"{api_url}/assess", files=files)
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {api_url}")
        return False
    finally:
        files['cv_file'].close()
    
    # Parse the response
    try:
        result = response.json()
    except json.JSONDecodeError:
        print("Error: Could not parse API response as JSON")
        return False
    
    # Print the results
    print("\n=== O-1A Visa Qualification Assessment ===")
    print(f"Qualification Rating: {result.get('qualification_rating', 'Unknown')}")
    print(f"Criteria Met: {result.get('criteria_met_count', 0)} out of 8")
    print("\nJustification:")
    print(result.get('rating_justification', 'No justification provided'))
    
    print("\nCriteria Matches:")
    criteria_matches = result.get('criteria_matches', {})
    for criterion, matches in criteria_matches.items():
        if matches:
            print(f"\n{criterion.upper()} ({len(matches)} matches):")
            for match in matches:
                print(f"- {match.get('text', '')}")
                print(f"  Explanation: {match.get('explanation', '')}")
    
    return True

if __name__ == "__main__":
    # Default values
    api_url = "http://localhost:8000"
    cv_file_path = "data/cv_samples/sample_cv.txt"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    if len(sys.argv) > 2:
        cv_file_path = sys.argv[2]
    
    # Run the test
    success = test_api(api_url, cv_file_path)
    sys.exit(0 if success else 1) 