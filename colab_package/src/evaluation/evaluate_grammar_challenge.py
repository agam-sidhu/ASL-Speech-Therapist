"""Convenience wrapper for the curated ASL grammar challenge set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the curated ASL grammar challenge set.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--challenge_set", default=str(PROJECT_ROOT / "data" / "eval" / "asl_grammar_challenge.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--show_examples", type=int, default=20)
    parser.add_argument("--show_failures_only", action="store_true")
    parser.add_argument("--output_json", default=None, help="Optional path for a JSON evaluation summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.evaluation.evaluate_translation import run_translation_evaluation

    args.dataset = None
    args.split = "all"
    args.val_split = None
    args.test_split = None
    args.seed = None
    run_translation_evaluation(args)


if __name__ == "__main__":
    main()
