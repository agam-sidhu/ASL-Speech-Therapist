import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { API_ENDPOINTS } from "../config/api";
import { buildSignChunksFromText, extractSignKeywords, getSignVideoEmbedUrl, isPlaceholderGloss } from "../utils/aslVideo";

export default function Practice({ sessionData, setSessionData }) {
  const navigate = useNavigate();
  const videoRef = useRef(null);
  const openCameraButtonRef = useRef(null);
  const cameraControlsRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const streamRef = useRef(null);
  const recordingTimerRef = useRef(null);

  const MIN_RECORD_SECONDS = 2;

  const [recording, setRecording] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [recordingWarning, setRecordingWarning] = useState("");
  const [analysisResult, setAnalysisResult] = useState(null);

  const rawInput = sessionData?.rawInput || "";

  const expectedSigns =
    sessionData?.aslPrediction?.predicted_gloss_tokens?.length > 0 &&
    sessionData.aslPrediction.predicted_gloss_tokens[0] !== "I"
      ? sessionData.aslPrediction.predicted_gloss_tokens
      : [rawInput.toUpperCase()];
  const inputDrivenSigns = !isPlaceholderGloss(expectedSigns)
    ? expectedSigns.slice(0, 5)
    : buildSignChunksFromText(rawInput, 5) || extractSignKeywords(rawInput, 5);
  const expectedText = !isPlaceholderGloss(expectedSigns)
    ? sessionData?.aslPrediction?.predicted_gloss_text || inputDrivenSigns.join(" ")
    : inputDrivenSigns.join(" / ");

  const startCamera = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
    streamRef.current = stream;
    videoRef.current.srcObject = stream;
  };
  const startRecording = async () => {
    if (!streamRef.current || isAnalyzing) return;

    // Reset state
    setRecordingWarning("");
    chunksRef.current = [];
    setRecordingSeconds(0);

    const preferredMimeType = "video/webm;codecs=vp9";
    const fallbackMimeType = "video/webm";

    const recorderOptions = MediaRecorder.isTypeSupported(preferredMimeType)
      ? { mimeType: preferredMimeType }
      : (MediaRecorder.isTypeSupported(fallbackMimeType)
          ? { mimeType: fallbackMimeType }
          : undefined);

    const recorder = recorderOptions
      ? new MediaRecorder(streamRef.current, recorderOptions)
      : new MediaRecorder(streamRef.current);

    mediaRecorderRef.current = recorder;

    // Capture chunks
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };

    // ✅ THIS RUNS AFTER RECORDING STOPS (correct place)
    recorder.onstop = async () => {
      clearInterval(recordingTimerRef.current);
      setIsAnalyzing(true);

      try {
        const uploadType = recorder.mimeType || "video/webm";
        const blob = new Blob(chunksRef.current, { type: uploadType });

        console.log("VIDEO SIZE:", blob.size);

        if (!blob.size) {
          throw new Error("Recorded video is empty.");
        }

        const formData = new FormData();
        formData.append("file", blob, "asl_recording.webm");

        // const videoResponse = await axios.post(API_ENDPOINTS.analyzeVideo, formData, {
        //   timeout: 600000,   // 10 minutes
        // });
        const videoResponse = await axios.post(API_ENDPOINTS.analyzeVideo, formData);

        console.log("VIDEO RESPONSE:", videoResponse.data);

        const predictedSigns = videoResponse.data.predicted_signs || [];
        const confidences = videoResponse.data.confidences || [];

        if (predictedSigns.length === 0) {
          setAnalysisResult({
            summary: "No signs detected",
            feedback: ["Try again. Make sure your hands are visible."]
          });
          return;
        }

        const bestSign = predictedSigns[0];
        const confidence = Math.round((confidences[0] || 0) * 100);

        setAnalysisResult({
          summary: `Detected: ${bestSign} (${confidence}%)`,
          confidenceScore: confidence,
          predictedSigns,
          feedback: [`Model thinks you signed "${bestSign}" with ${confidence}% confidence.`]
        });

      } catch (error) {
        console.error("ERROR:", error);

        setAnalysisResult({
          predictedSigns: [],
          feedback: ["Could not analyze this recording.", error.message],
          summary: "Video analysis failed."
        });
      } finally {
        setIsAnalyzing(false);
      }
    };

    // 🚀 START RECORDING
    recorder.start();
    setRecording(true);

    // ✅ START TIMER
    if (recordingTimerRef.current) {
      clearInterval(recordingTimerRef.current);
    }

    recordingTimerRef.current = setInterval(() => {
      setRecordingSeconds(prev => prev + 1);
    }, 1000);

    // ✅ AUTO STOP AFTER 3 SECONDS (VERY IMPORTANT)
    setTimeout(() => {
      if (mediaRecorderRef.current?.state === "recording") {
        mediaRecorderRef.current.stop();
        setRecording(false);
      }
    }, 3000);
  };
  const stopRecording = () => {
    if (recording && recordingSeconds < MIN_RECORD_SECONDS) {
      setRecordingWarning(`Please record for at least ${MIN_RECORD_SECONDS} seconds before stopping.`);
      return;
    }

    if (mediaRecorderRef.current && recording) {
      clearInterval(recordingTimerRef.current);
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  useEffect(() => {
    return () => {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const formatDuration = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainder = seconds % 60;
    return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
  };

  const formatConfidence = (value) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return "N/A";
    return `${Math.round(numeric * 100)}%`;
  };

  const handleTryAgain = async () => {
    setAnalysisResult(null);
    setRecordingWarning("");
    setRecordingSeconds(0);

    try {
      const resetForm = new FormData();
      resetForm.append("force_reload", "true");
      await axios.post(API_ENDPOINTS.resetVideoSession, resetForm);
    } catch (error) {
      // If reset endpoint is temporarily unavailable, still allow retry locally.
      console.warn("Could not reset backend video session:", error);
    }

    if (setSessionData) {
      setSessionData((previous) => {
        if (!previous) return previous;
        return {
          ...previous,
          analysisResult: null,
        };
      });
    }

    // Move the user back to the camera controls for the next attempt.
    requestAnimationFrame(() => {
      cameraControlsRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
      openCameraButtonRef.current?.focus();
    });
  };

  return (
    <div style={{ padding: 24, maxWidth: 1200, margin: "0 auto" }}>
      <h1>Practice the Signs</h1>

      <div style={{ marginBottom: 24 }}>
        <h3>Your input</h3>
        <p>{rawInput || "No input available yet."}</p>
      </div>

      <div style={{ marginBottom: 24 }}>
        <h3>Suggested ASL signs</h3>
        <p>{expectedText || "No prediction available yet."}</p>
      </div>

      <div style={{ marginBottom: 24 }}>
        <h3>Example videos</h3>
        <p>Use these examples to mirror the signs before you record yourself. Up to five videos are shown.</p>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16 }}>
          {inputDrivenSigns.map((sign, index) => {
            const embedUrl = getSignVideoEmbedUrl(sign);

            return (
              <div key={`${sign}-${index}`} style={{ border: "1px solid #ddd", borderRadius: 16, padding: 12, background: "#fff" }}>
                <h4 style={{ marginTop: 0 }}>{sign}</h4>
                <iframe
                  title={`Example video for ${sign}`}
                  src={embedUrl}
                  style={{ width: "100%", aspectRatio: "16 / 9", border: 0, borderRadius: 12 }}
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                />
                <a href={embedUrl} target="_blank" rel="noreferrer" style={{ display: "inline-block", marginTop: 8 }}>
                  Open sign page
                </a>
              </div>
            );
          })}
        </div>
      </div>

      <div ref={cameraControlsRef} style={{ display: "flex", gap: 12, marginBottom: 16 }}>
        <button ref={openCameraButtonRef} onClick={startCamera}>Open Camera</button>
        <button onClick={startRecording} disabled={recording || isAnalyzing}>Start Recording</button>
        <button onClick={stopRecording} disabled={!recording || recordingSeconds < MIN_RECORD_SECONDS || isAnalyzing}>Stop Recording</button>
        <button onClick={handleTryAgain} disabled={recording || isAnalyzing}>Try Again</button>
      </div>

      {isAnalyzing && (
        <p style={{ marginTop: 0, color: "#1a73e8", fontWeight: 600 }}>
          Analyzing recording... this can take up to 2 minutes.
        </p>
      )}

      <p style={{ marginTop: 0, marginBottom: 12, fontWeight: 600, color: recording ? "#b00020" : "#333" }}>
        Recording length: {formatDuration(recordingSeconds)}
      </p>
      {recordingWarning && <p style={{ marginTop: 0, color: "#b00020" }}>{recordingWarning}</p>}

      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{ width: 480, borderRadius: 12, background: "#000" }}
      />

      {analysisResult && (
        <div style={{ marginTop: 24 }}>
          <h2>Predicted Signs</h2>
          <p>
            {Array.isArray(analysisResult?.predictedSigns) &&
            analysisResult.predictedSigns.length > 0
              ? analysisResult.predictedSigns.join(" ")
              : "No signs predicted."}
          </p>

          {Array.isArray(analysisResult.predictedSigns) && analysisResult.predictedSigns.length > 0 && (
            <div style={{ marginBottom: 12 }}>
              <h3 style={{ marginBottom: 8 }}>Model confidence by sign</h3>
              {Array.isArray(analysisResult?.predictedSigns) &&
                analysisResult.predictedSigns.length > 0 && (
                  <div>
                    {analysisResult.predictedSigns.map((sign, idx) => (
                      <p key={idx}>
                        {sign}: {formatConfidence(analysisResult.confidences?.[idx])}
                      </p>
                    ))}
                  </div>
                )}
              {/* {analysisResult.predictedSigns.map((sign, idx) => (
                <p key={`${sign}-${idx}`} style={{ margin: "4px 0" }}>
                  {sign}: {formatConfidence(analysisResult.confidences?.[idx])}
                </p>
              ))} */}
            </div>
          )}

          {analysisResult?.summary && <p>{analysisResult.summary}</p>}

          <h2>Feedback</h2>
          {typeof analysisResult?.weightedScore === "number" && (
            <p>
              Weighted Score: {Math.round(analysisResult.weightedScore)}%
              {analysisResult.grade ? ` (Grade ${analysisResult.grade})` : ""}
            </p>
          )}
          {typeof analysisResult.accuracy === "number" && (
            <p>Base Accuracy: {Math.round(analysisResult.accuracy)}%</p>
          )}
          {typeof analysisResult.confidenceScore === "number" && (
            <p>Confidence Score: {Math.round(analysisResult.confidenceScore)}%</p>
          )}
          {Array.isArray(analysisResult?.feedback) &&
            analysisResult.feedback.map((item, idx) => (
              <p key={idx}>{item}</p>
          ))}

          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <button onClick={handleTryAgain} disabled={recording || isAnalyzing}>Try Again</button>
            <button onClick={() => navigate("/results", { state: analysisResult })}>
              Continue
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
