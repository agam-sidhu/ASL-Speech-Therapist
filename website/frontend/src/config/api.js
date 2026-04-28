// config/api.js

export const API_BASE = "http://127.0.0.1:8000";

export const API_ENDPOINTS = {
  textToAsl: `${API_BASE}/api/text-to-asl`,
  analyzeVideo: `${API_BASE}/api/analyze-video`,
  audioToAsl: `${API_BASE}/api/audio-to-asl`,
  fullFeedback: `${API_BASE}/api/full-feedback`,
};