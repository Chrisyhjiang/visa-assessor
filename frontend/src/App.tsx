import React, { useState, ChangeEvent, FormEvent } from "react";

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

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<AssessmentResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

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

  return (
    <div
      style={{ maxWidth: 600, margin: "40px auto", fontFamily: "sans-serif" }}
    >
      <h1>O-1A Visa Assessor</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          accept=".pdf,.docx,.txt"
          onChange={handleFileChange}
        />
        <button type="submit" disabled={loading} style={{ marginLeft: 10 }}>
          {loading ? "Assessing..." : "Upload & Assess"}
        </button>
      </form>
      {error && <div style={{ color: "red", marginTop: 20 }}>{error}</div>}
      {result && (
        <div style={{ marginTop: 30 }}>
          <h2>Assessment Result</h2>
          <div>
            <b>Qualification Rating:</b> {result.qualification_rating}
          </div>
          <div>
            <b>Overall Score:</b> {result.overall_score}
          </div>
          <div>
            <b>Explanation:</b> {result.explanation}
          </div>
          <div>
            <b>Recommendations:</b>
            <ul>
              {result.recommendations &&
                result.recommendations.map((rec, i) => <li key={i}>{rec}</li>)}
            </ul>
          </div>
          <div>
            <b>Criteria Matches:</b>
            <ul>
              {result.criteria_matches &&
                Object.entries(result.criteria_matches).map(
                  ([criterion, data]) => (
                    <li key={criterion}>
                      <b>{criterion}</b>: Score {data.score}
                      <ul>
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
  );
}

export default App;
