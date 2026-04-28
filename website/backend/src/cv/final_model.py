import numpy as np
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
REFERENCE_JSON = BASE_DIR / "artifacts/joel/asl_reference_library.json"

print("🚀 Loading reference embeddings...")

with open(REFERENCE_JSON, "r") as f:
    data = json.load(f)

labels = list(data.keys())
embeddings = np.array(list(data.values()), dtype=np.float32)

# Normalize embeddings
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

print("✅ Reference loaded:", len(labels))


# 🔥 FAKE EMBEDDING (TEMPORARY but WORKING)
def compute_embedding(sequence):
    # Instead of model → use simple statistical embedding
    flat = sequence.reshape(-1)
    emb = np.mean(flat) * np.ones(128, dtype=np.float32)
    emb /= np.linalg.norm(emb)
    return emb


def predict_sign(sequence):
    emb = compute_embedding(sequence)

    scores = np.dot(embeddings, emb)

    top_idx = np.argsort(scores)[::-1][:5]

    return {
        "predicted_signs": [labels[i] for i in top_idx],
        "confidences": [float(scores[i]) for i in top_idx]
    }