# O-1A Visa Qualification Assessment API

This API assesses how qualified a person is for an O-1A immigration visa based on their CV. It analyzes the CV against the 8 criteria defined for O-1A visas and provides a qualification rating.

## System Design

The system uses a combination of:

1. **BGE Embeddings for RAG (Retrieval-Augmented Generation)**: To provide relevant context about O-1A visa criteria when analyzing a CV.
2. **Qwen 1.5 0.5B Model**: A small, efficient language model to analyze the CV content against the O-1A criteria and generate the assessment.

### Architecture

```
┌─────────────┐     ┌───────────────┐     ┌─────────────────┐
│ CV Document │────▶│ CV Parser     │────▶│ Text Content    │
└─────────────┘     └───────────────┘     └────────┬────────┘
                                                   │
                                                   ▼
┌─────────────┐     ┌───────────────┐     ┌────────────────┐
│ Knowledge   │────▶│ RAG Service   │◀────│ Assessment     │
│ Base        │     │ (BGE)         │     │ Service (Qwen) │
└─────────────┘     └───────────────┘     └────────┬───────┘
                                                   │
                                                   ▼
                                          ┌────────────────┐
                                          │ API Response   │
                                          └────────────────┘
```

## Features

- Analyzes CVs in PDF or DOCX format
- Evaluates against all 8 O-1A criteria:
  - Awards
  - Membership
  - Press
  - Judging
  - Original contribution
  - Scholarly articles
  - Critical employment
  - High remuneration
- Provides:
  - List of evidence for each criterion
  - Confidence score for each criterion
  - Overall qualification rating (low, medium, high)
  - Explanation of the assessment

## Technical Implementation

- **FastAPI**: For the API framework
- **BGE Embeddings**: For semantic search in the knowledge base
- **Qwen 1.5 0.5B**: For CV analysis and assessment generation
- **FAISS**: For efficient vector storage and retrieval
- **LangChain**: For RAG implementation

## API Endpoints

### POST /assess-cv

Assesses a CV for O-1A visa qualification.

**Request**:

- Form data with a file upload (PDF or DOCX)

**Response**:

```json
{
  "criteria_matches": {
    "Awards": {
      "criterion": "Awards",
      "evidence": ["Award 1", "Award 2"],
      "confidence": 0.85
    },
    "Membership": {
      "criterion": "Membership",
      "evidence": ["Membership 1"],
      "confidence": 0.75
    },
    ...
  },
  "overall_rating": "medium",
  "explanation": "The applicant meets 4 out of 8 criteria with moderate confidence..."
}
```

### GET /health

Health check endpoint.

**Response**:

```json
{
  "status": "healthy"
}
```

## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   uvicorn app.main:app --reload
   ```

## Design Choices

### Why BGE Embeddings for RAG?

BGE (BAAI General Embeddings) models are efficient and perform well for semantic search. The "small" version provides a good balance between performance and speed, making it ideal for retrieving relevant information about O-1A criteria.

### Why Qwen 1.5 0.5B?

Qwen 1.5 0.5B is one of the smallest yet capable language models available. It provides:

- Fast inference speed
- Low memory requirements
- Sufficient reasoning capabilities for CV analysis
- Good performance on text analysis tasks

This makes it ideal for a responsive API that needs to analyze documents efficiently.

## Evaluation

The system's assessment quality can be evaluated by:

1. **Accuracy**: Compare the system's assessments with expert immigration consultants' assessments
2. **Consistency**: Check if similar CVs receive similar ratings
3. **Evidence Quality**: Evaluate if the evidence identified is relevant to each criterion
4. **Explanation Quality**: Assess if the explanations are clear and helpful

## Future Improvements

- Fine-tune the Qwen model on O-1A visa assessment data
- Add more detailed criteria explanations to the knowledge base
- Implement user feedback mechanism to improve the system
- Add support for more document formats
- Develop a web interface for easier interaction
