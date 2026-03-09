"""Interactive testing script for English-to-ASL translation model.

Run with:
    python test_examples.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models.inference import load_inference_bundle
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


def parse_args():
    parser = argparse.ArgumentParser(description="Test English-to-ASL translation interactively.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beam_width", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading model...")
    bundle = load_inference_bundle(args.checkpoint, device=args.device)
    print(f"✓ Model loaded: {bundle.model_name}")
    print(f"  Device: {args.device}")
    print(f"  Beam width: {args.beam_width}")
    print()

    print("English → ASL Gloss Translator")
    print("=" * 80)
    print("Enter English text and press Enter to translate to ASL gloss.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 80)

    while True:
        try:
            english = input("\nPlease enter your English text: ").strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break

        if not english:
            continue

        if english.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break

        # Import here to avoid issues if model fails to load
        from src.models.inference import predict_gloss

        prediction = predict_gloss(
            english,
            bundle,
            device=args.device,
            max_len=32,
            beam_width=args.beam_width,
        )

        print(f"\n✓ Translation:")
        print(f"   ASL Gloss: {prediction.predicted_gloss_text}")
        print(f"   Tokens: {prediction.predicted_gloss_tokens}")

        if prediction.empty_after_postprocess:
            print(f"   ⚠️  Warning: Output became empty after removing special tokens")


if __name__ == "__main__":
    main()
