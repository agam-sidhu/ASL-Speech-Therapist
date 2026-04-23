"""ASL Speech Therapy - Interactive GUI

Gradio-based web interface for the end-to-end pipeline:
  Audio Recording -> ASR (Whisper) -> Text Normalization -> ASL Gloss Translation

Install Gradio first:
    pip install gradio

Run:
    python app.py

Then open the URL shown in your terminal (usually http://localhost:7860).
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import numpy as np
from scipy.io.wavfile import write as wav_write

from src.utils.config import (
    DEFAULT_ASR_COMPUTE_TYPE,
    DEFAULT_ASR_DEVICE,
    DEFAULT_ASR_MODEL_SIZE,
    DEFAULT_CHECKPOINT_DIR,
)

# ── Globals (loaded once) ────────────────────────────────────────────────────

_inference_bundle = None
_whisper_model = None

DEFAULT_CHECKPOINT = str(Path(DEFAULT_CHECKPOINT_DIR) / "best_model.pt")


def get_inference_bundle(checkpoint_path: str = DEFAULT_CHECKPOINT, device: str = "cpu"):
    """Lazy-load the English->ASL translation model."""
    global _inference_bundle
    if _inference_bundle is None:
        from src.models.inference import load_inference_bundle
        _inference_bundle = load_inference_bundle(checkpoint_path, device=device)
    return _inference_bundle


def get_whisper_model():
    """Lazy-load the Whisper ASR model."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            DEFAULT_ASR_MODEL_SIZE,
            device=DEFAULT_ASR_DEVICE,
            compute_type=DEFAULT_ASR_COMPUTE_TYPE,
        )
    return _whisper_model


# ── Pipeline Functions ───────────────────────────────────────────────────────

def transcribe_audio(audio_path: str) -> dict:
    """Run Whisper ASR on an audio file."""
    model = get_whisper_model()
    segments, info = model.transcribe(audio_path, vad_filter=True)
    transcript = " ".join(seg.text.strip() for seg in segments).strip()
    return {
        "transcript": transcript,
        "language": info.language if info and info.language else "unknown",
    }


def normalize_transcript(text: str) -> str:
    """Clean up ASR output for the translation model."""
    from src.nlp.normalize_text import normalize_text
    result = normalize_text(text, remove_fillers=True)
    return result["clean_text"]


def translate_to_gloss(clean_text: str, beam_width: int = 3) -> dict:
    """Run the English->ASL transformer."""
    from src.models.inference import predict_gloss
    bundle = get_inference_bundle()
    prediction = predict_gloss(
        clean_text,
        bundle=bundle,
        device="cpu",
        max_len=32,
        beam_width=beam_width,
    )
    return {
        "gloss": prediction.predicted_gloss_text,
        "tokens": prediction.predicted_gloss_tokens,
    }


# ── Main Processing Function ────────────────────────────────────────────────

def process_audio(audio_input, beam_width: int = 3):
    """Full pipeline: audio -> transcript -> normalized -> ASL gloss.

    Returns: (raw_transcript, clean_text, gloss_output, performance_text)
    """
    if audio_input is None:
        return "", "", "", ""

    # Handle Gradio audio formats
    if isinstance(audio_input, tuple):
        sr, audio_data = audio_input
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        if np.issubdtype(audio_data.dtype, np.floating):
            audio_data = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        wav_write(tmp.name, sr, audio_data)
        audio_path = tmp.name
    else:
        audio_path = audio_input

    # Stage 1: ASR
    t0 = time.time()
    asr_result = transcribe_audio(audio_path)
    asr_time = time.time() - t0
    raw_transcript = asr_result["transcript"]

    if not raw_transcript.strip():
        return "(no speech detected)", "", "", ""

    # Stage 2: Normalization
    clean_text = normalize_transcript(raw_transcript)

    # Stage 3: Translation
    t1 = time.time()
    result = translate_to_gloss(clean_text, beam_width=beam_width)
    trans_time = time.time() - t1

    gloss_output = result["gloss"]
    timing_info = f"Whisper processing: {asr_time:.2f}s  |  Translation: {trans_time:.4f}s"

    return raw_transcript, clean_text, gloss_output, timing_info


def process_text_only(text_input: str, beam_width: int = 3):
    """Text-only mode: skip ASR, go straight to normalization + translation."""
    if not text_input or not text_input.strip():
        return "", "", ""

    clean_text = normalize_transcript(text_input)

    t0 = time.time()
    result = translate_to_gloss(clean_text, beam_width=beam_width)
    trans_time = time.time() - t0

    timing = f"Translation: {trans_time:.4f}s"
    return clean_text, result["gloss"], timing


# ── Custom CSS ───────────────────────────────────────────────────────────────

APP_CSS = """
/* ── Header styling ── */
.pipeline-header {
    text-align: center;
    margin-bottom: 0.5em;
}
.pipeline-header h1 {
    color: #1a56db !important;
}
.pipeline-header h3 {
    color: #333333 !important;
}

/* ── ASL gloss output: big blue bold text ── */
.gloss-output textarea {
    font-size: 1.4em !important;
    font-weight: bold !important;
    text-align: center !important;
    color: #1a56db !important;
}

/* ── Red record indicator dot (not the whole button) ── */
button[aria-label="Record"] div,
button[aria-label="Record"] span.recording-indicator,
button[aria-label="Record"]::before,
.record-icon, .mic-icon {
    color: #dc2626 !important;
}
/* The small colored circle inside the Record button */
button[aria-label="Record"] span[style],
button[aria-label="Record"] > span:first-child {
    background-color: #dc2626 !important;
}

/* ── Centered translate button ── */
.translate-btn-row {
    display: flex;
    justify-content: center;
    margin-top: 0.5em;
}
.translate-btn-row button {
    min-width: 300px;
}

/* ── Performance bar styling ── */
.performance-bar {
    margin-top: 0.25em;
    margin-bottom: 0.5em;
}
.performance-bar textarea, .performance-bar input {
    text-align: center !important;
    font-style: italic;
}

/* ── Equal height columns ── */
.equal-cols {
    align-items: stretch !important;
}
.equal-cols > .column {
    display: flex;
    flex-direction: column;
}

/* ── Translate button colors ── */
.translate-btn button {
    background-color: #1a56db !important;
    border-color: #1a56db !important;
    color: #ffffff !important;
}
.translate-btn button:hover {
    background-color: #1845b0 !important;
    border-color: #1845b0 !important;
}
"""


# ── Gradio UI ────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="ASL Speech Therapy") as app:

        gr.Markdown(
            """
            # ASL Speech Therapy
            ### Audio -> Speech Recognition -> Text Normalization -> ASL Gloss Translation
            """,
            elem_classes=["pipeline-header"],
        )

        with gr.Tabs():

            # ── Tab 1: Audio Pipeline ────────────────────────────
            with gr.Tab("Audio -> ASL Gloss", id="audio_tab"):

                with gr.Row(elem_classes=["equal-cols"]):
                    with gr.Column(scale=1):
                        audio_input = gr.Audio(
                            label="Record or Upload Audio",
                            sources=["microphone", "upload"],
                            type="numpy",
                        )
                        beam_slider = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3,
                            label="Beam Width",
                            info="1 = greedy (fast), 3-5 = beam search (better quality)",
                        )

                    with gr.Column(scale=1):
                        raw_transcript = gr.Textbox(
                            label="Stage 1: Speech Recognition (Whisper ASR)",
                            interactive=False,
                            placeholder="What Whisper hears...",
                        )
                        clean_text = gr.Textbox(
                            label="Stage 2: Text Normalization",
                            interactive=False,
                            placeholder="Cleaned text for translation...",
                        )
                        gloss_output = gr.Textbox(
                            label="Stage 3: ASL Gloss Output",
                            interactive=False,
                            placeholder="ASL gloss will appear here...",
                            elem_classes=["gloss-output"],
                        )

                # Performance + Translate button centered below both columns
                with gr.Row(elem_classes=["performance-bar"]):
                    timing_display = gr.Textbox(
                        label="Performance",
                        interactive=False,
                        visible=True,
                    )

                with gr.Row(elem_classes=["translate-btn-row"]):
                    run_btn = gr.Button(
                        "Translate to ASL",
                        variant="primary",
                        size="lg",
                        elem_classes=["translate-btn"],
                    )

                run_btn.click(
                    fn=process_audio,
                    inputs=[audio_input, beam_slider],
                    outputs=[raw_transcript, clean_text, gloss_output, timing_display],
                )

            # ── Tab 2: Text-Only Mode ────────────────────────────
            with gr.Tab("Text -> ASL Gloss", id="text_tab"):

                gr.Markdown("Type English text directly to see the ASL gloss translation (skips audio/ASR).")

                with gr.Row(elem_classes=["equal-cols"]):
                    with gr.Column(scale=1):
                        text_input = gr.Textbox(
                            label="English Text",
                            placeholder="Type an English sentence here...",
                            lines=2,
                        )
                        beam_slider_text = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3,
                            label="Beam Width",
                        )

                    with gr.Column(scale=1):
                        clean_text_only = gr.Textbox(
                            label="Normalized Text",
                            interactive=False,
                        )
                        gloss_output_text = gr.Textbox(
                            label="ASL Gloss Output",
                            interactive=False,
                            elem_classes=["gloss-output"],
                        )
                        timing_text = gr.Textbox(
                            label="Performance",
                            interactive=False,
                            elem_classes=["performance-bar"],
                        )

                with gr.Row(elem_classes=["translate-btn-row"]):
                    translate_btn = gr.Button(
                        "Translate",
                        variant="primary",
                        size="lg",
                        elem_classes=["translate-btn"],
                    )

                translate_btn.click(
                    fn=process_text_only,
                    inputs=[text_input, beam_slider_text],
                    outputs=[clean_text_only, gloss_output_text, timing_text],
                )

                # Quick demo sentences
                gr.Markdown("#### Try these examples:")
                examples = gr.Examples(
                    examples=[
                        ["What is your name?"],
                        ["Where is the bathroom?"],
                        ["I like to play sports."],
                        ["Today, I am going to school"],
                        ["She is my friend."],
                        ["I want to learn sign language!"],
                    ],
                    inputs=[text_input],
                )

            # ── Tab 3: About ─────────────────────────────────────
            with gr.Tab("About", id="about_tab"):
                gr.Markdown(
                    """
                    ## ASL Speech Therapy Pipeline

                    **University of Southern California**
                    CSCI-535: Multimodal Probabilistic Learning of Human Communication

                    ### Pipeline Architecture

                    ```
                    Audio Input
                        |
                        v
                    Audio Preprocessing (mono 16kHz)
                        |
                        v
                    Whisper ASR (speech -> text)
                        |
                        v
                    Text Normalization (lowercase, remove fillers)
                        |
                        v
                    Transformer Seq2Seq (English -> ASL Gloss)
                        |
                        v
                    ASL Gloss Output
                    ```

                    ### Technical Details

                    **ASR:** OpenAI Whisper (via faster-whisper) -- an end-to-end Transformer
                    that processes Mel spectrograms directly into text, bypassing traditional
                    HMM-based pipelines.

                    **Translation Model:** Custom Transformer encoder-decoder trained on
                    639 hand-crafted conversational English-ASL gloss pairs. Achieves 87.8%
                    accuracy on the validation set with beam search decoding.

                    **Key ASL Grammar Conventions:**
                    - Topic-comment structure (time/topic first)
                    - WH-question words at the end of sentences
                    - Compound words joined with hyphens (e.g., COME-BACK, ICE-CREAM)
                    - Function words (articles, prepositions) dropped

                    ### Team
                    - Kevin -- ASR integration & English-to-ASL translation
                    - Joel -- Computer vision / sign recognition
                    - Agam -- Sign-to-written translation
                    - Prianshu -- Pronunciation scoring
                    """
                )

    return app


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASL Speech Therapy GUI")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to model checkpoint")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    args = parser.parse_args()

    # Override default checkpoint if provided
    if args.checkpoint != DEFAULT_CHECKPOINT:
        DEFAULT_CHECKPOINT = args.checkpoint

    print("Loading models on first request (Whisper + Translation)...")
    print(f"Checkpoint: {args.checkpoint}")

    app = build_app()
    light_theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
    ).set(
        body_background_fill="#ffffff",
        block_background_fill="#f9fafb",
        block_label_background_fill="#1a56db",
        block_label_text_color="#ffffff",
        button_primary_background_fill="#1a56db",
        button_primary_text_color="#ffffff",
        button_primary_background_fill_hover="#1845b0",
    )
    app.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        theme=light_theme,
        css=APP_CSS,
    )
