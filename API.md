# O-1A Visa Assessor API Documentation

## Overview

The O-1A Visa Assessor API provides endpoints for analyzing CVs and assessing qualification for O-1A visa applications. The API uses advanced natural language processing to evaluate how well an applicant meets the various O-1A visa criteria.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. However, it does require an OpenAI API key to be set in the environment variables for the backend service to function.

## Endpoints

### 1. Root Endpoint

#### GET /

Returns basic information about the API.

**Response**

```json
{
  "message": "Welcome to O-1A Visa Assessor API. Use /docs for API documentation."
}
```

### 2. Assess Visa Application

#### POST /assess-visa

Upload a CV and receive a detailed assessment of O-1A visa qualification.

**Request**

- Method: POST
- Content-Type: multipart/form-data
- Body Parameter: cv_file (file)

**Supported File Formats**

- PDF (.pdf)
- Microsoft Word (.docx)
- Text (.txt)

**Response**

```json
{
  "qualification_rating": "high|medium|low",
  "overall_score": 0.85,
  "criteria_matches": {
    "awards_and_recognition": {
      "score": 0.75,
      "evidence": ["Evidence text 1", "Evidence text 2"]
    }
    // ... other criteria
  },
  "explanation": "Detailed explanation of the assessment",
  "recommendations": ["Recommendation 1", "Recommendation 2"],
  "agent_explanation": "USCIS officer-style explanation",
  "agent_recommendations": [
    "Officer recommendation 1",
    "Officer recommendation 2"
  ]
}
```

**Response Fields**

- `qualification_rating`: Overall assessment of O-1A qualification (high/medium/low)
- `overall_score`: Numerical score between 0 and 1
- `criteria_matches`: Detailed assessment for each O-1A criterion
  - `score`: Confidence score for the criterion (0-1)
  - `evidence`: List of supporting evidence from the CV
- `explanation`: Detailed explanation of the assessment
- `recommendations`: List of suggested improvements
- `agent_explanation`: USCIS officer perspective on the application
- `agent_recommendations`: Specific recommendations from USCIS officer perspective

**Error Responses**

- 400 Bad Request: Invalid file format
- 500 Internal Server Error: Processing error

## O-1A Visa Criteria

The API evaluates applications against the following criteria:

1. Awards and recognition
2. Membership in associations requiring outstanding achievement
3. Published material about the person
4. Participation as a judge of others' work
5. Original contributions of major significance
6. Authorship of scholarly articles
7. Critical employment in distinguished organizations
8. High salary or remuneration

## Rate Limiting

Currently, there are no rate limits implemented. However, the OpenAI API used by the backend may have its own rate limits.

## Error Handling

The API uses standard HTTP status codes and returns detailed error messages in the following format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

## Examples

### Example cURL Request

```bash
curl -X POST "http://localhost:8000/assess-visa" \
     -H "Content-Type: multipart/form-data" \
     -F "cv_file=@path/to/your/cv.pdf"
```

### Example Python Request

```python
import requests

url = "http://localhost:8000/assess-visa"
files = {
    'cv_file': ('cv.pdf', open('path/to/your/cv.pdf', 'rb'), 'application/pdf')
}

response = requests.post(url, files=files)
assessment = response.json()
```
