from __future__ import annotations

from importlib.resources import read_text
import os
import json
import subprocess
import shutil
import uuid
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

if TYPE_CHECKING:
    from src.cv.joel_recognizer import JoelASLRecognizer

from src.cv.joel_recognizer import JoelASLRecognizer


def interpolate_sequence(seq, target_len=100):
    if seq.shape[0] == 0:
        return np.zeros((target_len, 75, 3), dtype=np.float32)

    x_old = np.linspace(0, 1, seq.shape[0])
    x_new = np.linspace(0, 1, target_len)

    new_seq = np.zeros((target_len, 75, 3))

    for i in range(75):
        for j in range(3):
            new_seq[:, i, j] = np.interp(x_new, x_old, seq[:, i, j])

    return new_seq.astype(np.float32)

# ------------------------------------------------------------------------------
# APP SETUP
# ------------------------------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# MODEL PATHS
# ------------------------------------------------------------------------------

JOEL_MODEL_PATH = BASE_DIR / "artifacts" / "joel" / "best_graph_siamese_simease2000_lambdafree.keras"
DATA_DIR = PROJECT_DIR / "535 Project"

JOEL_535_DIR = PROJECT_DIR / "535 Project"
JOEL_WORKER_PYTHON = Path("C:/k3clean/Scripts/python.exe")
JOEL_WORKER_PYTHON = Path(os.getenv("ASL_JOEL_WORKER_PYTHON", "python"))

_joel_recognizer = None

def get_joel_recognizer():
    global _joel_recognizer

    if _joel_recognizer is None:
        print("🚀 Loading Joel model (first request only)...")
        _joel_recognizer = JoelASLRecognizer()
        print("✅ Model loaded")

    return _joel_recognizer


_kevin_bundle: Any | None = None

# ------------------------------------------------------------------------------
# REQUEST MODELS (FIX FOR NETWORK ERROR)
# ------------------------------------------------------------------------------

class TextRequest(BaseModel):
    text: str

# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------

def get_kevin_bundle() -> Any:
    from src.models.inference import load_inference_bundle

    global _kevin_bundle
    if _kevin_bundle is None:
        _kevin_bundle = load_inference_bundle(
            str(BASE_DIR / "checkpoints" / "best_model.pt"),
            device="cpu",
        )
    return _kevin_bundle

# ------------------------------------------------------------------------------
# TEXT → ASL (FIXED JSON INPUT)
# ------------------------------------------------------------------------------
@app.post("/api/text-to-asl")
def text_to_asl(req: TextRequest):
    text = req.text.lower().strip()

    OVERRIDE = {
        "hello": ["HELLO"],
        "hi": ["HELLO"],
        "hey": ["HELLO"],
        "thanks": ["THANK-YOU"],
        "thank you": ["THANK-YOU"],
        "yes": ["YES"],
        "no": ["NO"],
    }

    tokens = OVERRIDE.get(text, [text.upper()])

    print("TEXT INPUT:", text)
    print("TOKENS:", tokens)

    return {
        "clean_text": text,
        "predicted_gloss_tokens": tokens,
        "predicted_gloss_text": " ".join(tokens),
        "model_name": "fast-fallback",
    }

# ------------------------------------------------------------------------------
# AUDIO → ASL
# ------------------------------------------------------------------------------

@app.post("/api/audio-to-asl")
async def audio_to_asl(file: UploadFile = File(...)):
    file_id = uuid.uuid4().hex
    audio_path = TEMP_DIR / f"{file_id}.wav"

    with audio_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    from src.pipeline.run_audio_pipeline import run_pipeline

    args = type(
        "Args",
        (),
        {
            "mic": False,
            "audio_file": str(audio_path),
            "model_size": "base",
            "asr_device": "cpu",
            "compute_type": "int8",
            "checkpoint": str(BASE_DIR / "checkpoints" / "best_model.pt"),
            "device": "cpu",
            "max_len": 32,
            "keep_fillers": False,
            "use_fallback": False,
            "debug": False,
        },
    )

    return run_pipeline(args)

# ------------------------------------------------------------------------------
# VIDEO ANALYSIS (SIAMESE MODEL)
# ------------------------------------------------------------------------------

@app.post("/api/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    import tempfile, shutil

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

    with open(temp_video.name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print("📹 Video saved:", temp_video.name)

    recognizer = get_joel_recognizer()  # ✅ SAFE lazy load

    result = recognizer.analyze_video(
        temp_video.name,
        expected=req_text if available else None
    )

    print("🧠 Prediction:", result)

    return result
# ------------------------------------------------------------------------------
# FEEDBACK SYSTEM
# ------------------------------------------------------------------------------

@app.post("/api/full-feedback")
def full_feedback(
    expected: str = Form(default=""),
    predicted: str = Form(default=""),
    confidences: str | None = Form(default=None),
):
    expected_tokens = expected.split()
    predicted_tokens = predicted.split()

    parsed_confidences = []
    if confidences:
        try:
            parsed_confidences = json.loads(confidences)
        except:
            pass

    correct = sum(1 for i in range(min(len(expected_tokens), len(predicted_tokens)))
                  if expected_tokens[i] == predicted_tokens[i])

    accuracy = (correct / len(expected_tokens) * 100) if expected_tokens else 100

    confidence_score = (
        sum(parsed_confidences) / len(parsed_confidences) * 100
        if parsed_confidences else accuracy
    )

    weighted_score = 0.7 * accuracy + 0.3 * confidence_score

    return {
        "accuracy": accuracy,
        "confidenceScore": confidence_score,
        "weightedScore": weighted_score,
        "summary": f"Score: {weighted_score:.0f}%"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    