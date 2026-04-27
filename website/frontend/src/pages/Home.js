import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { API_ENDPOINTS } from "../config/api";

export default function Home({ setSessionData }) {
  const navigate = useNavigate();
  const [text, setText] = useState("");
  const [audioFile, setAudioFile] = useState(null);
  const [mode, setMode] = useState("text");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
  setLoading(true);

  try {
    let response;

    if (mode === "text") {
      console.log("➡️ Sending TEXT request to:", API_ENDPOINTS.textToAsl);

      response = await axios.post(API_ENDPOINTS.textToAsl, {
        text: text.trim(),
      });

    } else {
      const formData = new FormData();
      formData.append("file", audioFile);

      console.log("➡️ Sending AUDIO request to:", API_ENDPOINTS.audioToAsl);

      response = await axios.post(API_ENDPOINTS.audioToAsl, formData);
    }

    console.log("✅ RESPONSE:", response.data);

    setSessionData({
      inputMode: mode,
      rawInput: mode === "text" ? text : audioFile?.name,
      aslPrediction: response.data,
    });

    navigate("/practice");

  } catch (err) {
    console.error("❌ FULL ERROR:", err);

    if (err.response) {
      alert("Backend error: " + JSON.stringify(err.response.data));
    } else if (err.request) {
      alert("No response from backend. Check server.");
    } else {
      alert("Error: " + err.message);
    }
  }

  setLoading(false);
};

  return (
    <div style={{ padding: 24 }}>
      <h1>ASL Speech Therapist</h1>

      <div style={{ marginBottom: 16 }}>
        <button onClick={() => setMode("text")}>Type</button>
        <button onClick={() => setMode("audio")}>Record Audio</button>
      </div>

      {mode === "text" ? (
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type the English sentence..."
          rows={6}
          style={{ width: "100%" }}
        />
      ) : (
        <input
          type="file"
          accept="audio/*"
          onChange={(e) => setAudioFile(e.target.files[0])}
        />
      )}

      <div style={{ marginTop: 16 }}>
        <button
          disabled={loading || (mode === "text" && !text) || (mode === "audio" && !audioFile)}
          onClick={handleSubmit}
        >
          {loading ? "Processing..." : "Submit"}
        </button>
      </div>
    </div>
  );
}