"""Shared source-of-truth service for audio/text -> ASL gloss inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.asl.fallback_rules import fallback_text_to_gloss
from src.nlp.normalize_text import normalize_text
from src.utils.config import (
    DEFAULT_ASR_COMPUTE_TYPE,
    DEFAULT_ASR_DEVICE,
    DEFAULT_ASR_MODEL_SIZE,
    DEFAULT_RECORD_SECONDS,
)

if TYPE_CHECKING:
    from src.models.inference import InferenceBundle


def run_text_to_asl(
    text: str,
    *,
    bundle: InferenceBundle | None = None,
    checkpoint: str | None = None,
    device: str = "cpu",
    max_len: int = 32,
    beam_width: int = 1,
    debug: bool = False,
    use_fallback: bool = False,
    include_fallback_compare: bool = False,
    remove_fillers: bool = True,
) -> dict[str, Any]:
    """Translate English text into ASL-style gloss output."""
    normalization_result = normalize_text(text, remove_fillers=remove_fillers)
    clean_text = normalization_result["clean_text"]

    if use_fallback:
        fallback = fallback_text_to_gloss(normalization_result["tokens"])
        return {
            "input_mode": "text",
            "input_text": text,
            "clean_text": clean_text,
            "normalized_tokens": normalization_result["tokens"],
            "model_name": "fallback_rules",
            "empty_after_postprocess": len(fallback["predicted_gloss_tokens"]) == 0,
            **fallback,
        }

    if bundle is None:
        if checkpoint is None:
            raise ValueError("Provide either a loaded inference bundle or a checkpoint path.")
        from src.models.inference import load_inference_bundle

        bundle = load_inference_bundle(checkpoint, device=device)

    from src.models.inference import predict_gloss

    prediction = predict_gloss(
        text,
        bundle=bundle,
        device=device,
        max_len=max_len,
        debug=debug,
        beam_width=beam_width,
        normalization_result=normalization_result,
    )
    payload = {
        "input_mode": "text",
        "input_text": text,
        "normalized_tokens": normalization_result["tokens"],
        **prediction.to_dict(),
    }

    if include_fallback_compare:
        payload["fallback_compare"] = fallback_text_to_gloss(normalization_result["tokens"])

    return payload


def run_audio_to_asl(
    *,
    audio_file: str | None = None,
    use_microphone: bool = False,
    duration: float = DEFAULT_RECORD_SECONDS,
    model_size: str = DEFAULT_ASR_MODEL_SIZE,
    asr_device: str = DEFAULT_ASR_DEVICE,
    compute_type: str = DEFAULT_ASR_COMPUTE_TYPE,
    bundle: InferenceBundle | None = None,
    checkpoint: str | None = None,
    device: str = "cpu",
    max_len: int = 32,
    beam_width: int = 1,
    debug: bool = False,
    use_fallback: bool = False,
    keep_fillers: bool = False,
) -> dict[str, Any]:
    """Run the full audio -> ASR -> normalization -> gloss pipeline."""
    if use_microphone:
        from src.audio.record_audio import record_from_microphone

        audio_path = record_from_microphone(duration=duration)
    elif audio_file:
        audio_path = audio_file
    else:
        raise ValueError("Provide `audio_file` or set `use_microphone=True`.")

    from src.audio.asr import transcribe_audio
    from src.audio.preprocess_audio import preprocess_audio_to_mono16k

    processed_audio_path = preprocess_audio_to_mono16k(audio_path)
    asr_result = transcribe_audio(
        processed_audio_path,
        model_size=model_size,
        device=asr_device,
        compute_type=compute_type,
    )

    result = run_text_to_asl(
        asr_result["raw_transcript"],
        bundle=bundle,
        checkpoint=checkpoint,
        device=device,
        max_len=max_len,
        beam_width=beam_width,
        debug=debug,
        use_fallback=use_fallback,
        remove_fillers=not keep_fillers,
    )
    result.update(
        {
            "input_mode": "audio",
            "audio_path": audio_path,
            "processed_audio_path": processed_audio_path,
            "raw_transcript": asr_result["raw_transcript"],
            "language": asr_result["language"],
            "confidence": asr_result["confidence"],
        }
    )
    return result
