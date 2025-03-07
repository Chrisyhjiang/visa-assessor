# O-1A Visa Qualification Assessment System Design

This document outlines the design decisions and architecture of the O-1A Visa Qualification Assessment System.

## System Requirements

The system is designed to fulfill the following requirements:

1. Accept a CV document (PDF/DOCX) as input
2. Analyze the CV against the 8 O-1A visa criteria
3. Identify elements in the CV that match each criterion
4. Provide a qualification rating (Low, Medium, High)
5. Explain the reasoning behind the rating

## Architecture Overview

The system follows a modular architecture with the following components:

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌─────────────┐
│ CV Document │───▶│ Text Extract │───▶│ LLM Inference │───▶│ Response    │
│ (PDF/DOCX)  │    │ & Parse      │    │ (vLLM)        │    │ Formatter   │
└─────────────┘    └──────────────┘    └───────────────┘    └─────────────┘
                                               ▲
                                               │
                                        ┌───────────────┐
                                        │ Optional RAG  │
                                        │ (BGE Embedder)│
                                        └───────────────┘
                                               ▲
                                               │
                                        ┌───────────────┐
                                        │ O-1A Legal    │
                                        │ Documents     │
                                        └───────────────┘
```

### Components

1. **Document Parser**

   - Extracts text from PDF and DOCX files
   - Handles different document formats
   - Preserves text structure where possible

2. **LLM Inference Engine**

   - Uses Qwen 1.5B model via vLLM for efficient inference
   - Processes CV text with carefully designed prompts
   - Generates structured assessment output

3. **RAG System (Optional)**

   - Retrieves relevant legal context from O-1A documentation
   - Uses BGE embeddings for semantic search
   - Enhances the LLM's understanding of visa criteria

4. **Response Formatter**

   - Ensures consistent JSON output format
   - Handles error cases and edge conditions
   - Provides clean, structured responses

5. **FastAPI Application**
   - Provides RESTful API endpoints
   - Handles file uploads and validation
   - Manages the assessment workflow

## Design Decisions

### 1. Model Selection

We chose Qwen 1.5B as our base model for several reasons:

- **Size vs. Performance**: At 1.5B parameters, it's small enough to run on consumer hardware while still providing good performance.
- **Multilingual Support**: Qwen has strong multilingual capabilities, which is important for processing CVs from applicants of diverse backgrounds.
- **Structured Output**: The model performs well on tasks requiring structured outputs like JSON.

### 2. Inference Optimization

We use vLLM for inference because:

- **Efficiency**: vLLM provides significant speedups over standard Hugging Face inference.
- **Batching**: It handles batched requests efficiently.b
- **Memory Management**: It optimizes memory usage.
- **Continuous Batching**: It supports continuous batching for higher throughput.

### 3. RAG Implementation

Our RAG system is designed to:

- **Enhance Context**: Provide the LLM with relevant legal information about O-1A criteria.
- **Improve Accuracy**: Help the model make more accurate assessments by referencing official guidelines.
- **Reduce Hallucinations**: Ground the model's responses in factual information.

We use BGE embeddings because they:

- Perform well on semantic search tasks
- Are efficient to compute
- Work well with multilingual content

### 4. API Design

Our FastAPI implementation:

- **Simplicity**: Provides a clean, simple interface for CV assessment.
- **Robustness**: Handles errors gracefully and provides informative messages.
- **Flexibility**: Supports different modes of operation (with/without RAG).
- **Scalability**: Can be deployed on various platforms and scaled as needed.

## Evaluation Methodology

The system is designed to be evaluated on:

1. **Accuracy**: How well the system's assessments match expert assessments.
2. **Relevance**: How relevant the identified matches are to each criterion.
3. **Consistency**: How consistent the ratings are across similar CVs.
4. **Explainability**: How well the system explains its reasoning.

## Future Improvements

The system design allows for several future improvements:

1. **Model Upgrades**: Easily swap in larger or more capable models as they become available.
2. **Enhanced RAG**: Incorporate more sophisticated retrieval techniques.
3. **Multi-Document Support**: Process multiple documents per applicant (CV, cover letter, publications, etc.).
4. **User Interface**: Add a web interface for easier interaction.
5. **Feedback Loop**: Incorporate user feedback to improve the system over time.
6. **Model Fine-tuning**: Implement fine-tuning in the future to improve assessment accuracy.
