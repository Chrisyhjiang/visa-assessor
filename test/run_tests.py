"""
Run all tests for the O-1A Visa Qualification Assessment System.
"""

import os
import sys
import subprocess

# Add parent directory to path so we can import from it
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_tests():
    """Run all tests"""
    print("=== Running Tests for O-1A Visa Qualification Assessment System ===")
    
    # Get the directory of this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List of test files
    test_files = [
        "test_system.py",
        "test_api.py",
        "test_client.py"
    ]
    
    # Run each test
    for test_file in test_files:
        test_path = os.path.join(test_dir, test_file)
        if os.path.exists(test_path):
            print(f"\n=== Running {test_file} ===")
            result = subprocess.run(
                [sys.executable, test_path],
                cwd=test_dir
            )
            if result.returncode != 0:
                print(f"Test {test_file} failed with return code {result.returncode}")
                return result.returncode
        else:
            print(f"Test file {test_file} not found")
    
    print("\n=== All tests completed successfully ===")
    return 0

if __name__ == "__main__":
    sys.exit(run_tests()) 