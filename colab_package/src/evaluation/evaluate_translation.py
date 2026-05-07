"""Evaluate English->ASL translation with overlap and grammar-oriented metrics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate English->ASL translation quality.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--challenge_set", default=None, help="Optional curated grammar challenge JSON file.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beam_width", type=int, default=1)
    parser.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    parser.add_argument(
        "--val_split",
        type=float,
        default=None,
        help="Validation split ratio. Defaults to checkpoint split metadata when available.",
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=None,
        help="Test split ratio. Defaults to checkpoint split metadata when available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Split seed. Defaults to checkpoint split metadata when available.",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--show_examples", type=int, default=20)
    parser.add_argument("--output_json", default=None, help="Optional path to save evaluation summary JSON.")
    parser.add_argument(
        "--show_failures_only",
        action="store_true",
        help="Only print qualitative examples with non-exact predictions.",
    )
    return parser.parse_args()


def _resolve_split_parameters(args: argparse.Namespace) -> tuple[float, float, int]:
    """Resolve split config from CLI overrides or checkpoint metadata."""
    default_split = {"val_split": 0.15, "test_split": 0.0, "seed": 42}
    if args.split == "all" or args.challenge_set:
        return (
            default_split["val_split"] if args.val_split is None else args.val_split,
            default_split["test_split"] if args.test_split is None else args.test_split,
            default_split["seed"] if args.seed is None else args.seed,
        )

    import torch

    payload = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_split = payload.get("split_config", {})
    return (
        checkpoint_split.get("val_split", default_split["val_split"]) if args.val_split is None else args.val_split,
        checkpoint_split.get("test_split", default_split["test_split"]) if args.test_split is None else args.test_split,
        checkpoint_split.get("seed", default_split["seed"]) if args.seed is None else args.seed,
    )


def _load_eval_records(args: argparse.Namespace) -> tuple[list[dict[str, object]], str, dict[str, float | int] | None]:
    from src.data.dataset import load_paired_records
    from src.data.preprocess_dataset import preprocess_records
    from src.data.splits import select_split

    if args.challenge_set:
        challenge_path = Path(args.challenge_set)
        payload = json.loads(challenge_path.read_text(encoding="utf-8"))
        examples = payload.get("examples", payload)
        records: list[dict[str, object]] = []
        for item in examples:
            records.append(
                {
                    "english": str(item["english"]),
                    "gloss": str(item["gloss"]),
                    "challenge_categories": list(item.get("categories", [])),
                    "example_id": item.get("id"),
                    "source": item.get("source"),
                }
            )
        return records, f"challenge:{args.challenge_set}", None

    if not args.dataset:
        raise ValueError("Provide --dataset or --challenge_set.")

    val_split, test_split, seed = _resolve_split_parameters(args)
    records = load_paired_records(args.dataset)
    records = preprocess_records(records)
    records = select_split(
        records,
        split=args.split,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )
    return records, args.dataset, {"val_split": val_split, "test_split": test_split, "seed": seed}


def run_translation_evaluation(args: argparse.Namespace) -> None:
    from src.evaluation.translation_analysis import analyze_translation_case
    from src.models.inference import load_inference_bundle
    from src.services.asl_pipeline import run_text_to_asl
    from src.training.metrics import corpus_bleu

    records, dataset_label, split_config = _load_eval_records(args)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    if not records:
        raise ValueError("No records selected for evaluation.")

    bundle = load_inference_bundle(args.checkpoint, device=args.device)

    analyses = []
    all_refs: list[list[str]] = []
    all_hyps: list[list[str]] = []
    examples: list[dict[str, object]] = []
    for index, record in enumerate(records):
        result = run_text_to_asl(
            str(record["english"]),
            bundle=bundle,
            device=args.device,
            beam_width=args.beam_width,
        )
        analysis = analyze_translation_case(
            str(record["english"]),
            str(record["gloss"]),
            result["predicted_gloss_tokens"],
        )
        challenge_categories = list(record.get("challenge_categories", []))
        analysis_categories = challenge_categories or analysis["reference_categories"]
        analysis["english"] = record["english"]
        analysis["reference_gloss"] = record["gloss"]
        analysis["predicted_gloss"] = result["predicted_gloss_text"]
        analysis["analysis_categories"] = analysis_categories
        analysis["example_id"] = record.get("example_id")
        analyses.append(analysis)

        all_refs.append(analysis["reference_tokens"])
        all_hyps.append(analysis["predicted_tokens"])

        example_payload = {
            "english": record["english"],
            "reference": record["gloss"],
            "predicted": result["predicted_gloss_text"],
            "notes": analysis["notes"],
            "categories": analysis_categories,
            "reorder_required": analysis["reorder_required"],
            "exact_match": analysis["exact_match"],
            "bleu": round(analysis["bleu"], 4),
        }
        if (not args.show_failures_only or not analysis["exact_match"]) and len(examples) < args.show_examples:
            examples.append(example_payload)

    corpus_result = corpus_bleu(all_refs, all_hyps)
    exact_matches = sum(1 for item in analyses if item["exact_match"])
    avg_aligned_acc = sum(item["aligned_token_accuracy"] for item in analyses) / len(analyses)
    avg_token_f1 = sum(item["token_overlap_f1"] for item in analyses) / len(analyses)
    well_formed_rate = sum(1 for item in analyses if item["well_formed"]) / len(analyses)
    english_copy_rate = sum(1 for item in analyses if item["copies_english_order"]) / len(analyses)
    function_word_leak_rate = sum(1 for item in analyses if item["retained_function_words"]) / len(analyses)

    reorder_cases = [item for item in analyses if item["reorder_required"]]
    reorder_success_rate = (
        sum(1 for item in reorder_cases if item["follows_reference_order"]) / len(reorder_cases)
        if reorder_cases
        else 0.0
    )
    category_summary: dict[str, dict[str, object]] = {}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for analysis in analyses:
        for category in analysis["analysis_categories"]:
            grouped[category].append(analysis)
    for category, items in sorted(grouped.items()):
        category_reorder = [item for item in items if item["reorder_required"]]
        category_summary[category] = {
            "count": len(items),
            "exact_match_rate": round(sum(1 for item in items if item["exact_match"]) / len(items), 4),
            "avg_bleu": round(sum(item["bleu"] for item in items) / len(items), 4),
            "english_copy_rate": round(sum(1 for item in items if item["copies_english_order"]) / len(items), 4),
            "function_word_leak_rate": round(sum(1 for item in items if item["retained_function_words"]) / len(items), 4),
            "reorder_success_rate": round(
                sum(1 for item in category_reorder if item["follows_reference_order"]) / len(category_reorder),
                4,
            )
            if category_reorder
            else None,
        }

    print("\n" + "=" * 96)
    print("TRANSLATION EVALUATION")
    print("=" * 96)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Evaluation source: {dataset_label}")
    print(f"Split: {args.split} ({len(records)} samples)")
    print(f"Beam width: {args.beam_width}")
    if split_config is not None:
        print(f"Split config: {json.dumps(split_config, ensure_ascii=False)}")
    print(f"Corpus BLEU: {corpus_result['corpus_bleu']:.4f}")
    print(f"Exact match: {exact_matches}/{len(analyses)} ({100 * exact_matches / len(analyses):.1f}%)")
    print(f"Aligned token accuracy: {avg_aligned_acc:.4f}")
    print(f"Token overlap F1: {avg_token_f1:.4f}")
    print(f"Well-formed gloss rate: {well_formed_rate:.4f}")
    print(f"English-order copy rate: {english_copy_rate:.4f}")
    print(f"Function-word leak rate: {function_word_leak_rate:.4f}")
    print(
        f"Reorder-sensitive cases: {len(reorder_cases)} | Reference-order success: {reorder_success_rate:.4f}"
    )
    print("\nCategory breakdown:")
    for category, metrics in category_summary.items():
        print(f"- {category}: {json.dumps(metrics, ensure_ascii=False)}")

    print("\n" + "─" * 96)
    print(f"{'English':<24} {'Reference':<22} {'Predicted':<22} {'BLEU':<8} Categories | Notes")
    print("─" * 96)
    for example in examples:
        notes = ",".join(example["notes"]) if example["notes"] else "-"
        categories = ",".join(example["categories"]) if example["categories"] else "-"
        print(
            f"{example['english']:<24.24} {example['reference']:<22.22} "
            f"{example['predicted']:<22.22} {example['bleu']:<8.4f} {categories} | {notes}"
        )

    summary = {
        "dataset": dataset_label,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "total_samples": len(analyses),
        "beam_width": args.beam_width,
        "split_config": split_config,
        "corpus_bleu": round(corpus_result["corpus_bleu"], 4),
        "exact_match_rate": round(exact_matches / len(analyses), 4),
        "aligned_token_accuracy": round(avg_aligned_acc, 4),
        "token_overlap_f1": round(avg_token_f1, 4),
        "well_formed_rate": round(well_formed_rate, 4),
        "english_copy_rate": round(english_copy_rate, 4),
        "function_word_leak_rate": round(function_word_leak_rate, 4),
        "reorder_case_count": len(reorder_cases),
        "reorder_success_rate": round(reorder_success_rate, 4),
        "category_summary": category_summary,
    }
    print(f"\nJSON summary: {json.dumps(summary)}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "examples": examples,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"Saved JSON summary to {output_path}")


def main() -> None:
    run_translation_evaluation(parse_args())


if __name__ == "__main__":
    main()
