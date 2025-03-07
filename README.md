# O-1A Visa Qualification Assessment System

This system provides an API for assessing how qualified a person is for an O-1A immigration visa based on their CV. It analyzes the CV against the 8 criteria defined for O-1A visas and provides a qualification rating.

## Features

- CV text extraction from PDF, DOCX, and TXT files
- Analysis of CV content against O-1A criteria
- Qualification rating (Low, Medium, High)
- Detailed explanation of matches for each criterion
- Optional RAG (Retrieval-Augmented Generation) with legal documents

## System Architecture

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌─────────────┐
│ CV Document │───▶│ Text Extract │───▶│ LLM Inference │───▶│ Response    │
│ (PDF/DOCX)  │    │ & Parse      │    │ (vLLM/HF)     │    │ Formatter   │
└─────────────┘    └──────────────┘    └───────────────┘    └─────────────┘
                                               ▲
                                               │
                                        ┌───────────────┐
                                        │ Optional RAG  │
                                        └───────────────┘
```

## Project Structure

```
visa-assessor/
├── app.py                  # Main API application
├── model_setup.py          # Model initialization and inference
├── run.py                  # Script to run the application
├── requirements.txt        # Project dependencies
├── setup/                  # Setup scripts
│   ├── setup.py            # Main setup script
│   ├── setup_cpu.py        # CPU-only setup
│   ├── download_model.py   # Model download script
│   └── update_env.py       # Environment update script
├── test/                   # Test scripts
│   ├── test_system.py      # System tests
│   ├── test_api.py         # API tests
│   ├── test_client.py      # Client tests
│   └── run_tests.py        # Script to run all tests
├── processors/             # Processing modules
│   ├── document_parser.py  # CV document parsing
│   ├── prompt_builder.py   # Prompt construction
│   └── response_formatter.py # Response formatting
└── rag/                    # RAG components
    ├── document_processor.py # Document processing
    ├── create_vector_db.py   # Vector database creation
    └── retriever.py          # Retrieval functionality
```

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## Installation

### Installation

```bash
# Install dependencies
python setup/setup.py

# Download the model
python setup/download_model.py
```

### Running the Application

```bash
# Run the application
python run.py
```

### Testing

```bash
# Run all tests
python test/run_tests.py
```

## Usage

### Basic Usage

Run the API server:

```bash
python app.py
```

The API will be available at http://localhost:8000

### Test Mode

Run the API server in test mode (uses mock LLM responses):

```bash
python test_api.py
```

### Using RAG

1. First, prepare legal documents by placing them in the `data/legal_docs` directory.

2. Process the documents and create the vector database:

```bash
python -m rag.document_processor
python -m rag.create_vector_db
```

3. Run the API with RAG enabled:

```bash
USE_RAG=true python app.py
```

## API Endpoints

### GET /

Returns basic information about the API.

### POST /assess

Assesses O-1A visa qualification based on an uploaded CV.

**Request:**

- Form data with a file field named `cv_file`

**Response:**

```json
{
  "criteria_matches": {
    "awards": [...],
    "membership": [...],
    "press": [...],
    "judging": [...],
    "original_contribution": [...],
    "scholarly_articles": [...],
    "critical_employment": [...],
    "high_remuneration": [...]
  },
  "qualification_rating": "Medium",
  "rating_justification": "...",
  "criteria_met_count": 4
}
```

## Testing

You can test the API using the provided test client:

```bash
python test_client.py
```

This will send a sample CV to the API and display the results.

## Cloud Deployment

### Cloud Deployment

1. Create an account on a cloud provider and set up an instance.

2. SSH into the instance and clone the repository:

```bash
git clone https://github.com/yourusername/visa-assessor.git
cd visa-assessor
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the API:

```bash
python app.py
```

## Environment Variables

The system can be configured using the following environment variables:

- `USE_RAG`: Set to "true" to enable RAG (default: "false")
- `PORT`: Port to run the API on (default: 8000)

## Design Choices

### Model Selection

We chose Qwen 1.5B as our base model because:

- It's small enough to run on consumer hardware
- It has good multilingual support
- It performs well on structured tasks

### RAG Implementation

The RAG system uses:

- BGE embeddings for semantic search
- FAISS for efficient vector storage and retrieval
- Chunked legal documents for relevant context

## Evaluation

The system can be evaluated by:

1. **Accuracy**: Compare system assessments with expert assessments
2. **Relevance**: Evaluate the relevance of RAG-retrieved legal context
3. **Consistency**: Check consistency of ratings across similar CVs
4. **Explainability**: Assess the quality of explanations for each criterion match

## Future Work

In the future, we plan to implement model fine-tuning to improve the accuracy of the assessments. This will involve collecting expert assessments and using them to fine-tune the model.

## Performance Optimizations

The API has been optimized for better performance:

### Asynchronous Processing

The API now uses an asynchronous processing model:

1. When you submit a document, you immediately receive a request ID
2. You can check the status of your request using the `/assess/status/{request_id}` endpoint
3. This allows the server to process documents in the background without blocking the client

### Parallel Document Processing

PDF document processing has been optimized with parallel page extraction:

- Multiple pages are processed simultaneously using ThreadPoolExecutor
- This significantly improves performance for large PDF documents

### Response Caching

The model inference has been optimized with caching:

- Identical prompts will return cached results
- This reduces processing time for similar documents
- The cache uses an LRU (Least Recently Used) strategy to manage memory

### Client Example

A sample client implementation is provided in `docs/async_client_example.html` that demonstrates:

- How to submit documents to the asynchronous API
- How to poll for results
- How to display progress to users

### Usage Example

```python
import requests
import time
import json

# Submit a document
with open('my_cv.pdf', 'rb') as f:
    files = {'cv_file': f}
    response = requests.post('http://localhost:9000/assess', files=files)
    data = response.json()
    request_id = data['request_id']

# Poll for results
while True:
    status_response = requests.get(f'http://localhost:9000/assess/status/{request_id}')
    status_data = status_response.json()

    if status_data['status'] == 'completed':
        # Process completed
        result = status_data['result']
        print(json.dumps(result, indent=2))
        break
    elif status_data['status'] == 'error':
        # Error occurred
        print(f"Error: {status_data['error']}")
        break

    # Still processing, wait and try again
    print("Processing...")
    time.sleep(2)
```

## License

[MIT License](LICENSE)
