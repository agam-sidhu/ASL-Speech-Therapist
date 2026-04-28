"""Evaluate ASR output quality on labeled audio examples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ASR quality with WER/CER.")
    parser.add_argument("--manifest", required=True, help="JSON/JSONL/CSV with audio_path and reference_text fields.")
    parser.add_argument("--model_size", default="base")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute_type", default="int8")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--show_examples", type=int, default=10)
    return parser.parse_args()


def load_manifest(path: str) -> list[dict[str, str]]:
    """Load ASR evaluation records from JSON, JSONL, or CSV."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    records: list[dict[str, str]] = []

    if suffix == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("data", [])
        for item in payload:
            if "audio_path" in item and "reference_text" in item:
                records.append(
                    {
                        "audio_path": str(item["audio_path"]),
                        "reference_text": str(item["reference_text"]),
                    }
                )
    elif suffix == ".jsonl":
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if "audio_path" in item and "reference_text" in item:
                    records.append(
                        {
                            "audio_path": str(item["audio_path"]),
                            "reference_text": str(item["reference_text"]),
                        }
                    )
    elif suffix == ".csv":
        with file_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if "audio_path" in row and "reference_text" in row:
                    records.append(
                        {
                            "audio_path": str(row["audio_path"]),
                            "reference_text": str(row["reference_text"]),
                        }
                    )
    else:
        raise ValueError("Manifest must be .json, .jsonl, or .csv")

    if not records:
        raise ValueError("No usable ASR evaluation records found.")
    return records


def main() -> None:
    args = parse_args()

    from src.audio.asr import transcribe_audio
    from src.audio.preprocess_audio import preprocess_audio_to_mono16k
    from src.evaluation.asr_metrics import char_error_rate, word_error_rate
    from src.nlp.normalize_text import normalize_text

    records = load_manifest(args.manifest)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    wer_scores = []
    cer_scores = []
    examples: list[dict[str, object]] = []

    for index, record in enumerate(records):
        processed_audio_path = preprocess_audio_to_mono16k(record["audio_path"])
        asr_result = transcribe_audio(
            processed_audio_path,
            model_size=args.model_size,
            device=args.device,
            compute_type=args.compute_type,
        )
        hypothesis = asr_result["raw_transcript"]

        ref_norm = normalize_text(record["reference_text"], remove_fillers=False)
        hyp_norm = normalize_text(hypothesis, remove_fillers=False)

        wer = word_error_rate(ref_norm["tokens"], hyp_norm["tokens"])
        cer = char_error_rate(ref_norm["clean_text"], hyp_norm["clean_text"])
        wer_scores.append(wer)
        cer_scores.append(cer)

        if index < args.show_examples:
            examples.append(
                {
                    "audio_path": record["audio_path"],
                    "processed_audio_path": processed_audio_path,
                    "reference_text": record["reference_text"],
                    "hypothesis_text": hypothesis,
                    "normalized_reference": ref_norm["clean_text"],
                    "normalized_hypothesis": hyp_norm["clean_text"],
                    "wer": round(wer, 4),
                    "cer": round(cer, 4),
                }
            )

    avg_wer = sum(wer_scores) / len(wer_scores)
    avg_cer = sum(cer_scores) / len(cer_scores)

    print("\n" + "=" * 96)
    print("ASR EVALUATION")
    print("=" * 96)
    print(f"Manifest: {args.manifest}")
    print(f"Samples: {len(records)}")
    print(f"Model size: {args.model_size}")
    print(f"Average WER: {avg_wer:.4f}")
    print(f"Average CER: {avg_cer:.4f}")
    print("\n" + "─" * 96)
    for example in examples:
        print(f"Audio: {example['audio_path']}")
        print(f"Processed audio: {example['processed_audio_path']}")
        print(f"Reference: {example['reference_text']}")
        print(f"Hypothesis: {example['hypothesis_text']}")
        print(f"Normalized ref/hyp: {example['normalized_reference']} | {example['normalized_hypothesis']}")
        print(f"WER: {example['wer']:.4f} | CER: {example['cer']:.4f}")
        print("─" * 96)

    summary = {
        "manifest": args.manifest,
        "total_samples": len(records),
        "model_size": args.model_size,
        "average_wer": round(avg_wer, 4),
        "average_cer": round(avg_cer, 4),
    }
    print(f"JSON summary: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
