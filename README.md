# O-1A Visa Qualification Assessment API

This API assesses how qualified a person is for an O-1A immigration visa based on their CV. It analyzes the CV against the 8 criteria defined for O-1A visas and provides a qualification rating.

## System Design

The system uses a combination of:

1. **BGE Embeddings for RAG (Retrieval-Augmented Generation)**: To provide relevant context about O-1A visa criteria when analyzing a CV.
2. **Qwen2.5-0.5B Model**: A small, efficient language model to analyze the CV content against the O-1A criteria and generate the assessment.

### Architecture

```
┌─────────────┐     ┌───────────────┐     ┌─────────────────┐
│ CV Document │────▶│ CV Parser     │────▶│ Text Content    │
└─────────────┘     └───────────────┘     └────────┬────────┘
                                                   │
                                                   ▼
                                          ┌────────────────┐
                                          │ Assessment     │
                                          │ Service (Qwen) │
                                          └────────┬───────┘
                                                   │
                                                   ▼
┌─────────────┐     ┌───────────────┐     ┌────────────────┐
│ Knowledge   │────▶│ RAG Service   │────▶│ Criterion      │
│ Base        │     │ (BGE)         │     │ Information    │
└─────────────┘     └───────────────┘     └────────┬───────┘
                                                   │
                                                   ▼
                                          ┌────────────────┐
                                          │ API Response   │
                                          └────────────────┘
```

## Features

- Analyzes CVs in TXT format main, but also DOCX and PDF (untested)
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
- **Multilingual Support**: Capable of analyzing CVs in multiple languages thanks to Qwen2.5-0.5B's and BGE's strong multilingual capabilities

## Key Enhancements

The system has been enhanced to:

1. **Support All Fields of Employment**: The assessment system now recognizes achievements across diverse industries including Business, Arts, Medicine, Law, Finance, Technology, Education, and Government/Public Service.

2. **Direct Evidence Extraction**: The system directly extracts evidence for each criterion from the CV before falling back to LLM analysis, improving accuracy and reducing hallucinations.

3. **Specialized Criteria Handling**:

   - **Critical Employment**: Identifies STEM, government, military, and leadership positions with specialized keyword matching
   - **High Remuneration**: Detects salary information with specific thresholds ($150,000 to $350,000 for medium confidence, >$350,000 for high confidence)

4. **Improved Evidence Validation**: Filters out section titles, "N/A" values, and other non-evidence items to ensure only valid evidence is considered.

5. **Enhanced Confidence Scoring**: Calculates confidence scores based on the quality and quantity of evidence found for each criterion.

## Technical Implementation

- **FastAPI**: For the API framework
- **BGE Embeddings**: For semantic search in the knowledge base
- **Qwen2.5-0.5B**: For CV analysis and assessment generation
- **FAISS**: For efficient vector storage and retrieval
- **LangChain**: For RAG implementation

## API Endpoints

### POST /assess-cv

Assesses a CV for O-1A visa qualification.

**Request**:

- Form data with a file upload (PDF, DOCX, or TXT)

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
2. Create a virtual environment via the following steps:
   - Create virtual environment: `python -m venv venv`
   - Activate virtual environment:
     - Windows: `venv\Scripts\activate`
     - Unix/MacOS: `source venv/bin/activate`
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the application:
   ```
   python run.py
   ```
   or
   ```
   uvicorn app.main:app --reload
   ```

### Hardware Requirements

**For optimal performance speed, run this application on a machine with an NVIDIA A100 GPU.**

While the application can run on CPU or other GPUs, an A100 GPU will provide the fastest processing times for both the embedding generation and LLM inference steps. The Qwen2.5-0.5B model and BGE embeddings are optimized for GPU acceleration.

## Design Choices

### Why BGE Embeddings for RAG?

BGE (BAAI General Embeddings) models are efficient and perform well for semantic search. The system tries to use the largest available BGE model, falling back to smaller versions if necessary, providing a good balance between performance and speed.

BGE models also offer strong multilingual capabilities, supporting over 100 languages with consistent performance across language boundaries. This makes them ideal for processing international CVs without requiring translation.

### Why Qwen2.5-0.5B?

Qwen2.5-0.5B is one of the smallest yet capable language models available. It provides:

- Fast inference speed
- Low memory requirements
- Sufficient reasoning capabilities for CV analysis
- Good performance on text analysis tasks
- **Strong multilingual capabilities**: Supports analysis of CVs in multiple languages including English, Spanish, French, German, Chinese, and many others without requiring translation

This makes it ideal for a responsive API that needs to analyze documents efficiently.

**Please checkout DESIGN.md for more information on the Design chocies of this project"**

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
