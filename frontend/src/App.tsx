import React, { useState, ChangeEvent, FormEvent, useEffect } from "react";
import "./styles.css";

type CriteriaMatch = {
  score: number;
  evidence: string[];
};

type AssessmentResult = {
  qualification_rating: string;
  overall_score: number;
  explanation: string;
  recommendations: string[];
  criteria_matches: Record<string, CriteriaMatch>;
};

const App: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AssessmentResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [animateScore, setAnimateScore] = useState(false);

  useEffect(() => {
    if (result) {
      // Trigger animation after result is set
      setTimeout(() => {
        setAnimateScore(true);
      }, 300);
    } else {
      setAnimateScore(false);
    }
  }, [result]);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setResult(null);
      setError(null);
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!file) {
      setError("Please select a file.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    const formData = new FormData();
    formData.append("cv_file", file);
    try {
      const response = await fetch("http://localhost:8000/assess-visa", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Server error");
      }
      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Function to determine color based on score
  const getScoreColor = (score: number): string => {
    if (score >= 0.7) return "#4caf50";
    if (score >= 0.4) return "#ff9800";
    return "#f44336";
  };

  // Function to determine emoji based on score
  const getScoreEmoji = (score: number): string => {
    if (score >= 0.7) return "🌟";
    if (score >= 0.4) return "⭐";
    return "⚠️";
  };

  // Function to render circular progress
  const CircularProgress = ({
    percentage,
    color,
  }: {
    percentage: number;
    color: string;
  }) => {
    const radius = 40;
    const circumference = 2 * Math.PI * radius;
    const strokeDashoffset = circumference * (1 - percentage);

    return (
      <div className="circular-progress">
        <svg width="100" height="100" viewBox="0 0 100 100">
          <circle
            cx="50"
            cy="50"
            r={radius}
            fill="none"
            stroke="#e6e6e6"
            strokeWidth="8"
          />
          <circle
            className={animateScore ? "progress-circle-animated" : ""}
            cx="50"
            cy="50"
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth="8"
            strokeDasharray={circumference}
            strokeDashoffset={animateScore ? strokeDashoffset : circumference}
            strokeLinecap="round"
            transform="rotate(-90 50 50)"
          />
          <text
            x="50"
            y="50"
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize="20"
            fontWeight="bold"
            fill={color}
          >
            {(percentage * 100).toFixed(0)}%
          </text>
        </svg>
      </div>
    );
  };

  return (
    <div className="container">
      <header className="app-header">
        <h1 className="app-title">O-1A Visa Assessor</h1>
        <p className="app-subtitle">
          Upload your CV for an AI-powered assessment of your O-1A visa
          eligibility
        </p>
      </header>

      <div className="card">
        <div className="upload-section">
          <h2>Upload Your CV</h2>
          <p>We support PDF, DOCX, and TXT file formats</p>

          <form onSubmit={handleSubmit}>
            <div className="file-input-wrapper">
              <label className="file-input-label">
                <span className="button-icon">📄</span>
                Choose File
                <input
                  type="file"
                  accept=".pdf,.docx,.txt"
                  onChange={handleFileChange}
                  className="file-input"
                />
              </label>
            </div>

            {file && (
              <div className="file-name">
                Selected file: <strong>{file.name}</strong>
              </div>
            )}

            <button
              type="submit"
              disabled={loading || !file}
              className="button"
            >
              {loading ? (
                <>
                  <span className="button-icon">⏳</span>
                  Analyzing...
                </>
              ) : (
                <>
                  <span className="button-icon">🚀</span>
                  Assess Eligibility
                </>
              )}
            </button>
          </form>
        </div>

        {loading && (
          <div className="loading-spinner">
            <div className="spinner"></div>
          </div>
        )}

        {error && <div className="error-message">{error}</div>}

        {result && (
          <div className="results-section">
            <div className="results-header">
              <h2 className="results-title">Assessment Results</h2>
            </div>

            <div className="summary-grid">
              <div className="summary-card">
                <div className="summary-label">Qualification Rating</div>
                <div
                  className="summary-value"
                  style={{ color: getScoreColor(result.overall_score) }}
                >
                  {getScoreEmoji(result.overall_score)}{" "}
                  {result.qualification_rating}
                </div>
              </div>
              <div className="summary-card score-summary-card">
                <div className="summary-label">Overall Score</div>
                <CircularProgress
                  percentage={result.overall_score}
                  color={getScoreColor(result.overall_score)}
                />
              </div>
            </div>

            <div className="explanation-section">
              <h3>Explanation</h3>
              <p>{result.explanation}</p>
            </div>

            <div className="recommendations-section">
              <h3>Recommendations</h3>
              <ul>
                {result.recommendations &&
                  result.recommendations.map((rec, i) => (
                    <li key={i}>{rec}</li>
                  ))}
              </ul>
            </div>

            <div className="criteria-section">
              <h3>Criteria Analysis</h3>
              <ul className="criteria-list">
                {result.criteria_matches &&
                  Object.entries(result.criteria_matches).map(
                    ([criterion, data]) => (
                      <li key={criterion} className="criteria-item">
                        <div className="criteria-header">
                          <h4 className="criteria-name">{criterion}</h4>
                          <span
                            className="score-badge"
                            style={{
                              backgroundColor: getScoreColor(data.score),
                            }}
                          >
                            {getScoreEmoji(data.score)} Score:{" "}
                            {(data.score * 100).toFixed(0)}%
                          </span>
                        </div>
                        <ul className="evidence-list">
                          {data.evidence &&
                            data.evidence.map((ev, j) => <li key={j}>{ev}</li>)}
                        </ul>
                      </li>
                    )
                  )}
              </ul>
            </div>
          </div>
        )}
      </div>

      <footer className="app-footer">
        <p>© {new Date().getFullYear()} O-1A Visa Assessor - Powered by AI</p>
      </footer>
    </div>
  );
};

export default App;
