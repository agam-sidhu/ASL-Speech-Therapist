"""Evaluate a trained checkpoint on test sentences and compute BLEU scores.

Usage:
    python src/training/evaluate_checkpoint.py \
        --checkpoint checkpoints/best_model.pt \
        --dataset data/asl_gloss_pairs_v2.json \
        --beam_width 3
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--show_examples", type=int, default=20, help="Number of examples to print")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    from src.asl.postprocess_gloss import clean_gloss_tokens
    from src.data.dataset import load_paired_records
    from src.data.preprocess_dataset import preprocess_records
    from src.models.inference import load_inference_bundle
    from src.training.metrics import compute_bleu, corpus_bleu

    bundle = load_inference_bundle(args.checkpoint, device=args.device)
    records = load_paired_records(args.dataset)
    records = preprocess_records(records)

    if args.max_samples:
        records = records[: args.max_samples]

    all_refs: list[list[str]] = []
    all_hyps: list[list[str]] = []
    exact_matches = 0
    examples: list[dict] = []

    for i, rec in enumerate(records):
        english = rec["english"]
        ref_gloss = rec["gloss"]

        src_tokens = bundle.src_tokenizer.tokenize(english)
        src_ids = [bundle.src_vocab.bos_idx] + bundle.src_vocab.encode(src_tokens) + [bundle.src_vocab.eos_idx]
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=args.device)

        generated = bundle.model.generate(
            src_tensor,
            bundle.tgt_vocab.bos_idx,
            bundle.tgt_vocab.eos_idx,
            max_len=32,
            beam_width=args.beam_width,
        )
        gen_ids = generated.squeeze(0).tolist()
        raw_tokens = bundle.tgt_vocab.decode(gen_ids)
        pred_tokens = clean_gloss_tokens(raw_tokens)

        ref_tokens = [t.upper() for t in bundle.tgt_tokenizer.tokenize(ref_gloss)]

        all_refs.append(ref_tokens)
        all_hyps.append(pred_tokens)

        pred_text = " ".join(pred_tokens)
        ref_text = " ".join(ref_tokens)

        if pred_text == ref_text:
            exact_matches += 1

        per_bleu = compute_bleu(ref_tokens, pred_tokens)

        if i < args.show_examples:
            examples.append({
                "english": english,
                "reference": ref_text,
                "predicted": pred_text,
                "bleu": round(per_bleu["bleu"], 4),
                "match": pred_text == ref_text,
            })

    corpus_result = corpus_bleu(all_refs, all_hyps)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset} ({len(records)} samples)")
    print(f"Beam width: {args.beam_width}")
    print(f"\nCorpus BLEU: {corpus_result['corpus_bleu']:.4f}")
    print(f"Exact match accuracy: {exact_matches}/{len(records)} ({100*exact_matches/len(records):.1f}%)")
    print(f"Brevity penalty: {corpus_result['brevity_penalty']:.4f}")
    for n in range(1, 5):
        key = f"p{n}"
        if key in corpus_result:
            print(f"  {n}-gram precision: {corpus_result[key]:.4f}")

    print(f"\n{'─' * 80}")
    print(f"{'English':<35} {'Reference':<20} {'Predicted':<20} BLEU")
    print(f"{'─' * 80}")
    for ex in examples:
        marker = "✓" if ex["match"] else "✗"
        print(f"{marker} {ex['english']:<33} {ex['reference']:<20} {ex['predicted']:<20} {ex['bleu']:.2f}")

    print(f"{'─' * 80}")

    # Output JSON summary
    summary = {
        "corpus_bleu": round(corpus_result["corpus_bleu"], 4),
        "exact_match_rate": round(exact_matches / len(records), 4),
        "total_samples": len(records),
        "exact_matches": exact_matches,
        "beam_width": args.beam_width,
    }
    print(f"\nJSON summary: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
