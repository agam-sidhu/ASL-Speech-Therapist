# 🧠 ASL AI Translator & Feedback System

---
## 🚀 Features

### 🔤 Text → ASL

* Converts English sentences into ASL gloss
* Uses transformer-based model
* Includes fallback rules for robustness

---

### 🎥 Video → ASL Recognition

* User records ASL signs via webcam
* Uses **Joel’s Siamese Graph Neural Network**
* Extracts 3D pose + hand landmarks using MediaPipe
* Matches against WLASL dataset

---

### ⚡ Fast Inference (Optimized)

* Uses **precomputed embeddings (JSON)**
* Avoids slow runtime comparisons
* Real-time prediction (<2s)

---

### 📊 Feedback System

* Compares predicted vs expected signs
* Outputs:

  * Accuracy
  * Confidence score
  * Weighted score
  * Detailed feedback

---

## 🧠 Models Used

### 👨‍🔬 Joel’s Model (Vision)

* Graph Neural Network + Transformer
* Siamese similarity learning
* Trained on **WLASL dataset**

### 🤖 Kevin and Agam’s Model (NLP)

* English → ASL gloss transformer
* Uses paired text-gloss data
---

## 🏗️ Architecture

```
Frontend (React)
        ↓
FastAPI Backend
        ↓
-------------------------
|  Text Model (Kevin)   |
|  Video Model (Joel)   |
-------------------------
        ↓
Feedback Engine
```

---
## 📂 Models

Due to size limits, models may not be included.

Download and place:

```
backend/artifacts/joel/
backend/checkpoints/
```

Required files:

* `best_graph_siamese_simease2000_compat.keras`
* `wlasl_label_encoder_simease2000.pkl`
* `asl_reference_library.json`
* `best_model.pt`

---

## 🎯 How It Works

### Video Pipeline

1. Record 3-second video
2. Extract pose + hand landmarks (75×3)
3. Convert to 100-frame sequence
4. Generate embedding
5. Compare with reference embeddings
6. Return top matching ASL signs

---

## 🧪 Example Output

```
Predicted Signs: YES
Confidence: 87%
Feedback:
- Good attempt 👍
- Slight improvement needed in hand orientation
```

---

## 🔥 Key Optimizations

* Precomputed embedding search (JSON)
* No runtime dataset scanning
* Lazy model loading
* Efficient cosine similarity matching

---

## 📌 Future Improvements

* Real-time live ASL recognition
* Sentence-level sign detection
---

