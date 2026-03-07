"""Prepare ASL translation data from Hugging Face datasets.

Default target dataset: achrafothman/aslg_pc12
Expected HF fields: text, gloss
Output format: english, gloss
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess_dataset import preprocess_records
from src.utils.config import DEFAULT_ASLG_DATASET_NAME, DEFAULT_ASLG_OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ASLG-PC12 dataset from Hugging Face.")
    parser.add_argument("--dataset_name", default=DEFAULT_ASLG_DATASET_NAME)
    parser.add_argument("--output_dir", default=str(DEFAULT_ASLG_OUTPUT_DIR))
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_save_json",
        action="store_true",
        help="Do not save processed train/val JSON files locally.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional limit for quick dry runs.",
    )
    return parser.parse_args()


def split_records(
    records: list[dict[str, str]],
    val_split: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Deterministically split records into train and validation."""
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1")

    shuffled = records[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - val_split))
    split_idx = max(1, min(split_idx, len(shuffled) - 1))

    train_records = shuffled[:split_idx]
    val_records = shuffled[split_idx:]
    return train_records, val_records


def main() -> None:
    args = parse_args()

    from datasets import load_dataset

    dataset = load_dataset(args.dataset_name)

    if "train" in dataset:
        source_split = dataset["train"]
    else:
        first_split_name = next(iter(dataset.keys()))
        source_split = dataset[first_split_name]

    mapped_records: list[dict[str, str]] = []
    for row in source_split:
        if "text" not in row or "gloss" not in row:
            continue

        mapped_records.append(
            {
                "english": str(row["text"]),
                "gloss": str(row["gloss"]),
            }
        )

    if args.max_samples is not None:
        mapped_records = mapped_records[: args.max_samples]

    if not mapped_records:
        raise ValueError("No usable records found in HF dataset.")

    cleaned_records = preprocess_records(mapped_records)
    train_records, val_records = split_records(cleaned_records, val_split=args.val_split, seed=args.seed)

    summary = {
        "dataset_name": args.dataset_name,
        "total_records": len(cleaned_records),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "seed": args.seed,
        "val_split": args.val_split,
    }

    if not args.no_save_json:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = output_dir / "train.json"
        val_path = output_dir / "val.json"

        with train_path.open("w", encoding="utf-8") as handle:
            json.dump(train_records, handle, indent=2, ensure_ascii=False)
        with val_path.open("w", encoding="utf-8") as handle:
            json.dump(val_records, handle, indent=2, ensure_ascii=False)

        summary["train_path"] = str(train_path)
        summary["val_path"] = str(val_path)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
