import requests
import os
import sys
import unittest
from fastapi.testclient import TestClient

# Add the parent directory to the path so we can import the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

class TestO1AAssessmentAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
    
    def test_health_endpoint(self):
        """Test that the health endpoint returns a 200 status code."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})
    
    def test_assess_cv_endpoint_invalid_format(self):
        """Test that the assess-cv endpoint rejects invalid file formats."""
        # Create a test file with an invalid extension
        with open("test_invalid.txt", "w") as f:
            f.write("This is a test CV")
        
        with open("test_invalid.txt", "rb") as f:
            response = self.client.post(
                "/assess-cv",
                files={"file": ("test_invalid.txt", f, "text/plain")}
            )
        
        # Clean up the test file
        os.remove("test_invalid.txt")
        
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file format", response.json()["detail"])

if __name__ == "__main__":
    unittest.main() 