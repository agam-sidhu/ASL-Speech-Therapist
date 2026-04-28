"""Small callable wrapper around the trained text-to-gloss model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.models.inference import InferenceBundle, load_inference_bundle
from src.services.asl_pipeline import run_text_to_asl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GLOSS_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "project_finetune_v2_v4_contrastive" / "best_model.pt"
)


@lru_cache(maxsize=2)
def _load_bundle(checkpoint_path: str, device: str) -> InferenceBundle:
    return load_inference_bundle(checkpoint_path, device=device)


def translate_text_to_gloss(
    text: str,
    *,
    checkpoint_path: str | Path = DEFAULT_GLOSS_CHECKPOINT,
    device: str = "cpu",
    beam_width: int = 3,
) -> str:
    """Translate an English text string into an ASL gloss string."""
    bundle = _load_bundle(str(checkpoint_path), device)
    result = run_text_to_asl(
        text,
        bundle=bundle,
        device=device,
        beam_width=beam_width,
    )
    return str(result["predicted_gloss_text"])
