"""Audit paired English->ASL gloss data for grammar signal and leakage risk."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit translation dataset grammar signal.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show_examples", type=int, default=8)
    parser.add_argument("--output", default=None, help="Optional path to save JSON audit summary.")
    return parser.parse_args()


def load_paired_records_lightweight(path: str) -> list[dict[str, str]]:
    """Load English/gloss pairs without importing training dependencies."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    records: list[dict[str, str]] = []

    if suffix == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("data", [])
        for item in payload:
            if "english" in item and "gloss" in item:
                records.append({"english": str(item["english"]), "gloss": str(item["gloss"])})
    elif suffix == ".jsonl":
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if "english" in item and "gloss" in item:
                    records.append({"english": str(item["english"]), "gloss": str(item["gloss"])})
    elif suffix == ".csv":
        with file_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if "english" in row and "gloss" in row:
                    records.append({"english": str(row["english"]), "gloss": str(row["gloss"])})
    else:
        raise ValueError("Unsupported dataset format. Use .json, .jsonl, or .csv.")

    if not records:
        raise ValueError("No usable paired records found in dataset.")
    return records


def main() -> None:
    args = parse_args()

    from src.data.splits import split_records
    from src.evaluation.translation_analysis import (
        classify_reference_categories,
        coarse_gloss_template,
        english_tokens_for_analysis,
        reorder_strength,
    )
    from src.nlp.normalize_text import normalize_text

    records: list[dict[str, str]] = []
    for record in load_paired_records_lightweight(args.dataset):
        clean_english = normalize_text(record["english"], remove_fillers=False)["clean_text"]
        clean_gloss = " ".join(record["gloss"].strip().upper().split())
        if clean_english and clean_gloss:
            records.append({"english": clean_english, "gloss": clean_gloss})
    train_records, val_records, test_records = split_records(
        records,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
    )

    src_lengths = []
    tgt_lengths = []
    src_vocab: set[str] = set()
    tgt_vocab: set[str] = set()
    category_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    target_start_counts: Counter[str] = Counter()
    gloss_counts: Counter[str] = Counter()
    sample_by_category: dict[str, list[dict[str, object]]] = defaultdict(list)

    for index, record in enumerate(records):
        english_tokens = english_tokens_for_analysis(record["english"])
        gloss_tokens = [token.upper() for token in record["gloss"].split()]
        categories = classify_reference_categories(record["english"], record["gloss"])
        template = coarse_gloss_template(record["gloss"])

        src_lengths.append(len(english_tokens))
        tgt_lengths.append(len(gloss_tokens))
        src_vocab.update(english_tokens)
        tgt_vocab.update(gloss_tokens)
        category_counts.update(categories)
        template_counts[template] += 1
        gloss_counts[" ".join(gloss_tokens)] += 1
        if gloss_tokens:
            target_start_counts[gloss_tokens[0]] += 1

        for category in categories:
            if len(sample_by_category[category]) < args.show_examples:
                sample_by_category[category].append(
                    {
                        "index": index,
                        "english": record["english"],
                        "gloss": record["gloss"],
                        "reorder_strength": reorder_strength(english_tokens, gloss_tokens),
                        "template": template,
                    }
                )

    def normalized_english(record: dict[str, str]) -> str:
        return " ".join(english_tokens_for_analysis(record["english"]))

    def normalized_gloss(record: dict[str, str]) -> str:
        return " ".join(token.upper() for token in record["gloss"].split())

    def split_overlap(reference_split: list[dict[str, str]], candidate_split: list[dict[str, str]]) -> dict[str, int]:
        ref_english = {normalized_english(record) for record in reference_split}
        ref_gloss = {normalized_gloss(record) for record in reference_split}
        ref_templates = {coarse_gloss_template(record["gloss"]) for record in reference_split}
        return {
            "english_exact_overlap": sum(1 for record in candidate_split if normalized_english(record) in ref_english),
            "gloss_exact_overlap": sum(1 for record in candidate_split if normalized_gloss(record) in ref_gloss),
            "template_overlap": sum(
                1 for record in candidate_split if coarse_gloss_template(record["gloss"]) in ref_templates
            ),
        }

    summary = {
        "dataset": args.dataset,
        "total_examples": len(records),
        "split_sizes": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "source_length": {
            "avg": round(statistics.mean(src_lengths), 3),
            "median": statistics.median(src_lengths),
            "min": min(src_lengths),
            "max": max(src_lengths),
        },
        "target_length": {
            "avg": round(statistics.mean(tgt_lengths), 3),
            "median": statistics.median(tgt_lengths),
            "min": min(tgt_lengths),
            "max": max(tgt_lengths),
        },
        "vocabulary": {
            "source": len(src_vocab),
            "target": len(tgt_vocab),
        },
        "category_counts": dict(category_counts.most_common()),
        "top_target_starts": target_start_counts.most_common(15),
        "top_templates": template_counts.most_common(15),
        "repeated_gloss_forms": [(gloss, count) for gloss, count in gloss_counts.items() if count > 1][:15],
        "split_overlap_risk": {
            "val_vs_train": split_overlap(train_records, val_records),
            "test_vs_train": split_overlap(train_records, test_records),
        },
        "sample_examples": sample_by_category,
    }

    print("\n" + "=" * 96)
    print("TRANSLATION DATASET AUDIT")
    print("=" * 96)
    print(f"Dataset: {args.dataset}")
    print(f"Examples: {summary['total_examples']}")
    print(
        f"Split sizes: train={summary['split_sizes']['train']} val={summary['split_sizes']['val']} "
        f"test={summary['split_sizes']['test']}"
    )
    print(
        f"Source length avg/median: {summary['source_length']['avg']} / {summary['source_length']['median']}"
    )
    print(
        f"Target length avg/median: {summary['target_length']['avg']} / {summary['target_length']['median']}"
    )
    print(
        f"Vocabulary size: source={summary['vocabulary']['source']} target={summary['vocabulary']['target']}"
    )
    print(f"Category counts: {json.dumps(summary['category_counts'], ensure_ascii=False)}")
    print(f"Split overlap risk: {json.dumps(summary['split_overlap_risk'], ensure_ascii=False)}")

    for category, examples in sample_by_category.items():
        print(f"\n[{category}]")
        for example in examples[: args.show_examples]:
            print(f"- #{example['index']}: {example['english']} => {example['gloss']}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved JSON summary to {output_path}")


if __name__ == "__main__":
    main()
