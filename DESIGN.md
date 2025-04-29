# O-1A Visa Assessor: Design Document

## System Design Overview

This document outlines the design choices, architecture, and evaluation methodology for the O-1A Visa Assessor application. The application uses machine learning techniques to analyze CVs and provide an assessment of an individual's qualification for an O-1A visa.

## Architecture

### 1. Technology Stack

- **FastAPI**: A modern, high-performance web framework for building APIs
- **Sentence Transformers**: ML models for encoding text into semantic vectors
- **PyPDF2/python-docx**: Libraries for text extraction from PDF and DOCX files
- **Scikit-learn**: For cosine similarity calculation and other ML utilities
- **NLTK**: For natural language processing tasks like sentence tokenization

### 2. Component Overview

The system is organized into the following primary components:

1. **API Layer**: Handles HTTP requests, file uploads, and JSON responses
2. **Document Processing**: Extracts and preprocesses text from uploaded CV files
3. **ML-based Criteria Matching**: Uses sentence transformers to match CV content to O-1A criteria
4. **Scoring and Rating**: Determines qualification ratings based on criteria matches
5. **Response Formatting**: Structures the assessment results for API responses

### 3. Data Flow

1. User uploads CV through the `/assess-visa` endpoint
2. System validates the file format (PDF, DOCX, TXT)
3. Text is extracted from the CV and split into sentences
4. Sentences are encoded into vectors using the sentence transformer model
5. System computes similarity between CV sentences and pre-defined criteria
6. Top matching sentences for each criterion are identified
7. Overall qualification scores and rating are calculated
8. Structured response with evidence and scores is returned to the user

## Machine Learning Approach

### 1. Why Sentence Transformers?

Traditional keyword or regex-based approaches have significant limitations:

- They can only match exact terms or patterns
- They miss semantically similar content expressed differently
- They cannot understand context or meaning
- They require complex rule systems to handle variations

Our sentence transformer approach addresses these issues by:

- Capturing semantic meaning rather than just keywords
- Understanding variations in how criteria might be expressed
- Providing similarity scores that reflect confidence levels
- Learning from examples without explicit rules

### 2. Model Selection

We chose the `paraphrase-MiniLM-L6-v2` model because it:

- Is optimized for semantic textual similarity tasks
- Has a reasonable size (80MB) for deployment
- Provides good performance with limited computational resources
- Handles a wide range of general topics and domains
- Has been fine-tuned on paraphrase data, making it good at recognizing semantic equivalence

### 3. Hybrid Matching Strategy

Our system uses two types of reference content for matching:

1. **Criteria Descriptions**: Formal definitions of each O-1A criterion
2. **Criteria Examples**: Curated examples of statements that would satisfy each criterion

This hybrid approach combines:

- **Description matching**: High-level semantic understanding
- **Example matching**: More nuanced pattern recognition

We weight example matches higher (70%) than description matches (30%) as they better represent actual criteria patterns.

### 4. Batch Processing for Scalability

For large documents, we:

- Process text in batches of 300 sentences
- Merge and consolidate results
- Take top-k sentences overall

This approach:

- Prevents memory issues with large documents
- Maintains performance on limited hardware
- Allows processing arbitrarily large CVs

## Scoring and Rating Methodology

### 1. Sentence Matching

For each sentence in the CV:

1. Compute embedding vector using transformer model
2. Calculate cosine similarity to criteria descriptions (weighted 30%)
3. Calculate cosine similarity to criteria examples (weighted 70%)
4. Combine weighted scores for overall similarity
5. Apply threshold (0.6) to filter low-confidence matches
6. Keep top k (5) sentences for each criterion

### 2. Criteria Scoring

For each criterion:

1. Calculate weighted average of top sentence scores
2. Scale by number of matches found (more matches = higher confidence)
3. Produce a final score between 0 and 1

### 3. Qualification Rating

Overall qualification is rated as:

- **High**: 3+ criteria with scores > 0.7 OR overall score > 0.65
- **Medium**: 1+ criteria with scores > 0.7 OR 3+ criteria with scores > 0.6 OR overall score > 0.5
- **Low**: Less than the above

This rating system aims to approximate the USCIS requirement of meeting at least 3 of the 8 criteria, while accounting for strength of evidence.

## Evaluation Approach

### 1. Qualitative Evaluation

The system should be evaluated by:

- Testing with real CV examples from known O-1A applicants
- Comparing system ratings with actual visa outcomes
- Expert review of evidence identification
- User feedback on assessment accuracy

### 2. Quantitative Metrics

Potential metrics include:

- **Precision**: Accuracy of identified evidence
- **Recall**: Ability to find all relevant evidence
- **F1 Score**: Balanced measure of precision and recall
- **Rating Accuracy**: Percentage of correct qualification ratings

### 3. Limitations and Edge Cases

The system has several known limitations:

- Dependency on CV comprehensiveness and formatting
- Limited understanding of specialized terminology
- Inability to verify truthfulness of claims
- No consideration of supporting documentation beyond the CV
- No access to actual USCIS evaluation criteria details

## Future Enhancements

1. **Model Fine-tuning**: Train on actual successful O-1A applications
2. **Entity Recognition**: Add specialized NER for identifying awards, organizations, etc.
3. **Domain Adaptation**: Create separate models for different fields (science, arts, business)
4. **Structured Data Extraction**: Extract structured information like education, work history
5. **Recommendation Engine**: Suggest improvements to strengthen applications
6. **UI Development**: Create user interface for easier interaction and visualization
7. **Multi-document Analysis**: Process supporting documents beyond the CV

## Conclusion

The ML-based O-1A Visa Assessor provides a more sophisticated approach than traditional keyword matching systems. By using semantic similarity with transformer models, the system can better understand and evaluate the complex criteria for O-1A visa qualification.

This design balances technical sophistication with practical deployability, creating a system that provides meaningful preliminary assessments while acknowledging the inherent limitations of automated immigration qualification analysis.
