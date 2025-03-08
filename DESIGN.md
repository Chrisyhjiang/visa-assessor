# O-1A Visa Assessment System Design Document

## Overview

This document explains the design choices made in implementing the O-1A Visa Qualification Assessment API. The system is designed to analyze a person's CV against the 8 criteria for O-1A visa qualification and provide an assessment of their eligibility.

## System Architecture

The system follows a modular architecture with the following components:

1. **API Layer (FastAPI)**: Handles HTTP requests and responses
2. **CV Parser**: Extracts text from PDF and DOCX files
3. **RAG System**: Provides relevant information about O-1A criteria
4. **Assessment Service**: Analyzes CVs against criteria and generates assessments
5. **Knowledge Base**: Contains information about each O-1A criterion

## Key Design Decisions

### 1. Choice of Embedding Model: BGE-small-en-v1.5

**Decision**: Use BAAI's BGE-small-en-v1.5 for the RAG system's embeddings.

**Rationale**:

- **Efficiency**: The small variant (384 dimensions) is computationally efficient
- **Performance**: Despite its size, it performs well on semantic search tasks
- **Balance**: Provides a good balance between accuracy and speed
- **Language Support**: Well-suited for English text analysis

**Alternatives Considered**:

- **Larger BGE models**: Would provide slightly better accuracy but at a significant performance cost
- **OpenAI embeddings**: Would require API calls, adding latency and external dependencies
- **MPNet**: Good performance but larger and slower than BGE-small

### 2. Choice of LLM: Qwen 1.5 0.5B

**Decision**: Use Qwen 1.5 0.5B as the language model for CV analysis.

**Rationale**:

- **Size**: At only 0.5B parameters, it's one of the smallest capable LLMs available
- **Speed**: Provides fast inference, essential for a responsive API
- **Efficiency**: Low memory requirements (can run on CPU if needed)
- **Capability**: Despite its small size, it has sufficient reasoning capabilities for CV analysis
- **Open Source**: Freely available and can be deployed without API costs

**Alternatives Considered**:

- **Larger Qwen models**: Would provide better analysis but at the cost of speed
- **Phi-2**: Similar size but slightly less capable for complex reasoning
- **Mistral 7B**: Much more capable but significantly larger and slower
- **API-based models**: Would add external dependencies and costs

### 3. RAG Implementation

**Decision**: Implement a Retrieval-Augmented Generation system using FAISS for vector storage.

**Rationale**:

- **Context Enhancement**: Provides the LLM with relevant information about O-1A criteria
- **Knowledge Base**: Allows storing detailed information about each criterion
- **Efficiency**: FAISS provides fast vector similarity search
- **Accuracy**: Improves the quality of assessments by providing specific criteria details

**Implementation Details**:

- Each criterion has a dedicated knowledge base file with detailed information
- Documents are chunked and embedded using BGE embeddings
- When analyzing a CV, relevant criterion information is retrieved and provided to the LLM

### 4. Assessment Methodology

**Decision**: Analyze each criterion separately and then combine results for an overall assessment.

**Rationale**:

- **Modularity**: Each criterion can be assessed independently
- **Explainability**: Provides clear evidence for each criterion
- **Accuracy**: Focused analysis on each criterion improves accuracy
- **O-1A Requirements**: Aligns with the actual visa requirement of meeting at least 3 criteria

**Implementation Details**:

- For each criterion, retrieve relevant information from the knowledge base
- Analyze the CV against each criterion separately
- Determine overall rating based on:
  - Number of criteria met (at least 3 required for O-1A)
  - Confidence scores for each criterion
  - Quality of evidence found

## Performance Considerations

1. **Model Loading**: The Qwen model is loaded once at service initialization to avoid repeated loading costs

2. **Chunking Strategy**: CV text is split into manageable chunks to:

   - Avoid token limits
   - Focus analysis on relevant sections
   - Improve processing efficiency

3. **Vector Search Optimization**: FAISS is used for efficient similarity search

4. **Asynchronous Processing**: FastAPI's asynchronous capabilities are leveraged for handling concurrent requests

## Evaluation Methodology

The system's assessment quality can be evaluated by:

1. **Expert Comparison**: Compare the system's assessments with those of immigration experts
2. **Precision and Recall**: Measure how accurately the system identifies relevant evidence
3. **Consistency**: Test with similar CVs to ensure consistent ratings
4. **User Feedback**: Collect feedback from users on assessment quality

## Limitations and Future Improvements

### Current Limitations

1. **Model Size**: The small Qwen model has limited reasoning capabilities compared to larger models
2. **Language Support**: Currently optimized for English CVs only
3. **Document Formats**: Limited to PDF and DOCX formats
4. **No Fine-tuning**: Models are not specifically fine-tuned for O-1A assessment

### Future Improvements

1. **Model Fine-tuning**: Fine-tune the Qwen model on O-1A assessment data
2. **Expanded Knowledge Base**: Add more detailed information about each criterion
3. **Multi-modal Support**: Add capability to analyze images and charts in CVs
4. **User Feedback Loop**: Implement a feedback mechanism to improve assessments
5. **Multilingual Support**: Add support for CVs in multiple languages
