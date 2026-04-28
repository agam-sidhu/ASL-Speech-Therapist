import React from "react";
import { useLocation } from "react-router-dom";

export default function Results({ sessionData }) {
  const location = useLocation();
  const resultData = location.state || sessionData?.analysisResult || sessionData || {};
  const feedback = resultData?.feedback || [];
  const feedbackItems = resultData?.feedbackItems || [];
  const summary = resultData?.summary || "";
  const weightedScore = resultData?.weightedScore;
  const confidenceScore = resultData?.confidenceScore;
  const accuracy = resultData?.accuracy;
  const grade = resultData?.grade;

  const speakFeedback = (text) => {
    if (!text || !window.speechSynthesis) return;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utterance);
  };

  const speakableText = resultData?.speakableFeedback || [summary, ...feedback].filter(Boolean).join(". ");

  return (
    <div style={{ padding: 24 }}>
      <h1>Results</h1>

      {summary && <p>{summary}</p>}

      {(typeof weightedScore === "number" || typeof accuracy === "number" || typeof confidenceScore === "number") && (
        <div style={{ marginBottom: 16 }}>
          {typeof weightedScore === "number" && (
            <p>
              Weighted Score: {Math.round(weightedScore)}%
              {grade ? ` (Grade ${grade})` : ""}
            </p>
          )}
          {typeof accuracy === "number" && <p>Base Accuracy: {Math.round(accuracy)}%</p>}
          {typeof confidenceScore === "number" && <p>Confidence Score: {Math.round(confidenceScore)}%</p>}
        </div>
      )}

      <h3>Feedback</h3>
      {feedbackItems.length > 0 ? feedbackItems.map((item, idx) => (
        <p key={idx}>
          {item.status === "correct" ? "Matched" : item.status === "missing" ? "Missing" : item.status === "extra" ? "Extra" : "Check"}: {item.message}
        </p>
      )) : feedback.map((item, idx) => (
        <p key={idx}>{item}</p>
      ))}

      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        <button onClick={() => speakFeedback(speakableText)}>Speak Feedback</button>
        {summary && <button onClick={() => speakFeedback(summary)}>Speak Summary</button>}
      </div>
    </div>
  );
}