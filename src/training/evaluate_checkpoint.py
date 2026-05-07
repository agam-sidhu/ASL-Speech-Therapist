"""Backward-compatible wrapper for translation evaluation.

Usage:
    python src/training/evaluate_checkpoint.py \
        --checkpoint checkpoints/project_finetune_v2_v4_contrastive/best_model.pt \
        --dataset data/active/project_finetune_v2_v4_contrastive.json \
        --beam_width 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint on dataset.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--show_examples", type=int, default=20, help="Number of examples to print")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.evaluation.evaluate_translation import run_translation_evaluation

    run_translation_evaluation(args)


if __name__ == "__main__":
    main()
