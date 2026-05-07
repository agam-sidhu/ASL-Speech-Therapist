"""Demo CLI for the trained English text -> ASL gloss model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "project_finetune_v2_v4_contrastive" / "best_model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate English text to ASL gloss.")
    parser.add_argument("--text", required=True, help="English text to translate.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beam_width", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    from src.services.gloss_inference import translate_text_to_gloss

    gloss = translate_text_to_gloss(
        args.text,
        checkpoint_path=checkpoint,
        device=args.device,
        beam_width=args.beam_width,
    )
    print(f"Input: {args.text}")
    print(f"ASL Gloss: {gloss}")


if __name__ == "__main__":
    main()
