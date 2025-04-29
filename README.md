# O-1A Visa Assessor

An ML-based FastAPI application that assesses a person's qualifications for an O-1A immigration visa based on their CV.

## System Design

The O-1A Visa Assessor application employs a machine learning approach to analyze CVs and determine qualification for O-1A visas. The system architecture consists of:

1. **Web API Layer** - FastAPI application that provides RESTful endpoints
2. **ML Processing Layer** - Sentence transformer models for semantic text analysis
3. **Document Processing** - Text extraction and processing utilities
4. **Semantic Matching** - Vector-based similarity matching against O-1A criteria
5. **Rating System** - ML-based scoring algorithm for visa qualification assessment

### ML-Based Approach

Unlike basic regex or keyword matching systems, this application uses:

1. **Sentence Transformers** - Pre-trained models that encode sentences into dense vector representations
2. **Semantic Similarity** - Cosine similarity to measure the semantic relatedness between CV content and O-1A criteria
3. **Example-Based Learning** - Using curated examples of each criterion to improve matching accuracy
4. **Weighted Scoring** - More sophisticated scoring that considers both quantity and quality of matches
5. **Batch Processing** - Efficient handling of large documents through batch processing

### Flow

1. User uploads CV file (PDF, DOCX, or TXT) to the API endpoint
2. API extracts text from the CV document
3. Text is split into meaningful sentences
4. Sentence transformer model converts sentences into vector embeddings
5. System computes similarity between CV sentences and pre-defined criteria descriptions/examples
6. Top matching sentences for each criterion are identified based on similarity score
7. Overall qualification scoring is calculated using weighted criteria matches
8. Response with criteria matches, evidence, and qualification rating is returned

## O-1A Criteria

The O-1A visa has 8 criteria, and applicants must satisfy at least 3 to qualify:

1. **Awards** - Receipt of nationally or internationally recognized prizes/awards for excellence
2. **Membership** - Membership in associations requiring outstanding achievement
3. **Press** - Published material about the person in professional publications
4. **Judging** - Participation as a judge of the work of others in the field
5. **Original Contribution** - Original scientific, scholarly, or business-related contributions of major significance
6. **Scholarly Articles** - Authorship of scholarly articles in professional journals or major media
7. **Critical Employment** - Employment in a critical or essential capacity at distinguished organizations
8. **High Remuneration** - Command of a high salary or remuneration

## Installation & Setup

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/visa-assessor.git
   cd visa-assessor
   ```

2. Create a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Run the application:

   ```
   python main.py
   ```

5. Access the API at http://localhost:8000/docs

## API Usage

### Endpoint: POST /assess-visa

Upload a CV file and get an assessment for O-1A visa qualification.

**Request:**

- Form data with file upload field "cv_file"

**Response:**

```json
{
  "criteria_matches": {
    "awards": {
      "score": 0.85,
      "evidence": ["List of matching sentences from CV"]
    },
    "membership": {
      "score": 0.72,
      "evidence": ["List of matching sentences from CV"]
    },
    ...
  },
  "qualification_rating": "high",
  "overall_score": 0.76
}
```

## Design Choices & Evaluation Methodology

### 1. Sentence Transformer Selection

We chose the `paraphrase-MiniLM-L6-v2` model for its balance of performance and efficiency. This model:

- Is trained specifically for semantic similarity tasks
- Performs well with limited computational resources
- Has a reasonable size for deployment scenarios
- Provides good similarity detection across various domains

### 2. Hybrid Matching Strategy

The system uses both criteria descriptions and examples:

- **Description Matching** - Ensures basic understanding of the criteria
- **Example Matching** - Provides more nuanced matching by showing the model what good matches look like
- **Weighted Combination** - Examples are weighted higher (70%) than descriptions (30%) as they better represent actual criteria

### 3. Batch Processing for Large Documents

For CVs with many sentences:

- Text is processed in batches of 300 sentences
- Results are merged and rescored
- This allows handling of arbitrarily large documents without memory issues

### 4. Scoring Methodology

The scoring system employs:

- **Threshold-Based Filtering** - Only sentences with similarity above 0.6 are considered
- **Top-K Selection** - Limited to top 5 sentences per criterion to focus on strongest evidence
- **Weighted Scoring** - Higher similarity matches receive greater weight
- **Scaling by Evidence Count** - Scores are scaled by the number of pieces of evidence found

### 5. Qualification Rating Logic

- **High Rating**: 3+ criteria with scores > 0.7 OR overall score > 0.65
- **Medium Rating**: 1+ criteria with scores > 0.7 OR 3+ criteria with scores > 0.6 OR overall score > 0.5
- **Low Rating**: Less than 2 criteria with sufficient evidence

## Limitations

1. The assessment is preliminary and not a substitute for legal advice
2. Text extraction may not capture all information from complex CV formats
3. The pre-trained embedding model may not perfectly understand specialized domain vocabulary
4. The system cannot verify the truthfulness of claims in the CV
5. Performance depends on the quality and comprehensiveness of the CV

## Future Improvements

1. Fine-tune embedding models on actual O-1A application data
2. Implement more sophisticated document structure analysis
3. Add entity recognition to better identify organizations, awards, etc.
4. Include counterfactual explanations to help improve applications
5. Extend with a UI for easier interaction and visualization of results
