import axios from "axios";
import { useState } from "react";
import { API_ENDPOINTS } from "../config/api";

export default function Feedback({ data }) {
  const [result, setResult] = useState(null);

  const run = async () => {
    const expectedTokens = data?.expectedTokens || data?.predicted_gloss_tokens || [];
    const predictedTokens = data?.predictedTokens || data?.predicted_gloss_tokens || [];

    const formData = new FormData();
    formData.append("expected", expectedTokens.join(" "));
    formData.append("predicted", predictedTokens.join(" "));

    const res = await axios.post(API_ENDPOINTS.fullFeedback, formData);
    setResult(res.data);
  };

  return (
    <div>
      <button onClick={run}>Analyze</button>

      {result?.summary && <p>{result.summary}</p>}
      {(result?.items || result?.feedback || []).map((item, i) => (
        <p key={i}>{typeof item === "string" ? item : item.message}</p>
      ))}
    </div>
  );
}