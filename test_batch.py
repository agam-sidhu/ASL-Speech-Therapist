"""Batch testing script with predefined English-ASL examples.

Run with:
    python test_batch.py --checkpoint checkpoints/best_model.pt

Or test specific categories:
    python test_batch.py --checkpoint checkpoints/best_model.pt --category questions
    python test_batch.py --checkpoint checkpoints/best_model.pt --category emotions
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.models.inference import load_inference_bundle, predict_gloss
from src.training.metrics import compute_bleu
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Test dataset organized by category
TEST_CASES: Dict[str, List[Tuple[str, str]]] = {
    "greetings": [
        ("hello", "HELLO"),
        ("hi", "HELLO"),
        ("goodbye", "GOODBYE"),
        ("good morning", "GOOD MORNING"),
        ("good night", "GOOD NIGHT"),
        ("how are you", "HOW YOU"),
    ],

    "basic_requests": [
        ("please help me", "PLEASE HELP ME"),
        ("can you help me", "YOU HELP ME CAN"),
        ("i need help", "I NEED HELP"),
        ("thank you", "THANK-YOU"),
        ("sorry", "SORRY"),
    ],

    "questions": [
        ("what is that", "THAT WHAT"),
        ("where is the bathroom", "BATHROOM WHERE"),
        ("when will you come", "YOU COME WHEN"),
        ("why are you sad", "YOU SAD WHY"),
        ("how old are you", "YOU AGE WHAT"),
    ],

    "emotions": [
        ("i feel happy", "I FEEL HAPPY"),
        ("i feel sad", "I FEEL SAD"),
        ("i am excited", "I EXCITED"),
        ("i am tired", "I TIRED"),
        ("i am scared", "I SCARED"),
    ],

    "learning": [
        ("i want to learn asl", "I WANT LEARN ASL"),
        ("i am studying asl", "I STUDY ASL"),
        ("do you understand", "YOU UNDERSTAND YOU"),
        ("i do not understand", "I NOT UNDERSTAND"),
        ("can you sign slower", "YOU SIGN SLOW CAN"),
    ],

    "daily_activities": [
        ("i go to school every day", "EVERY-DAY I GO SCHOOL"),
        ("i like to read books", "I LIKE READ BOOK"),
        ("i am hungry", "I HUNGRY"),
        ("i want water", "I WANT WATER"),
        ("what time is it", "TIME WHAT"),
    ],

    "family_and_relationships": [
        ("my mother is a doctor", "MY MOTHER DOCTOR"),
        ("i have two brothers", "I HAVE BROTHER TWO"),
        ("she is my friend", "SHE MY FRIEND"),
        ("they are my classmates", "THEY MY CLASSMATE"),
        ("i like you", "I LIKE YOU"),
    ],

    "advanced": [
        ("can you help me today", "TODAY YOU HELP ME CAN"),
        ("i will come back later", "LATER I COME-BACK"),
        ("the weather is nice today", "TODAY WEATHER NICE"),
        ("i graduated from college", "I FINISH COLLEGE"),
        ("let us practice signing", "WE PRACTICE SIGN"),
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Batch test English-to-ASL translation.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--category", default=None,
                       help=f"Test category ({', '.join(TEST_CASES.keys())}). If not specified, test all.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beam_width", type=int, default=3)
    parser.add_argument("--show_all", action="store_true", help="Show all results, not just failures")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading model...")
    bundle = load_inference_bundle(args.checkpoint, device=args.device)
    print(f"✓ Model loaded: {bundle.model_name}")
    print()

    # Select test cases
    if args.category:
        if args.category not in TEST_CASES:
            print(f"Error: Category '{args.category}' not found.")
            print(f"Available: {', '.join(TEST_CASES.keys())}")
            sys.exit(1)
        test_data = {args.category: TEST_CASES[args.category]}
    else:
        test_data = TEST_CASES

    # Run tests
    total_correct = 0
    total_tests = 0
    all_bleu_scores = []

    for category, examples in test_data.items():
        print(f"\n{'=' * 90}")
        print(f"Category: {category.upper()}")
        print(f"{'=' * 90}")
        print(f"{'English':<40} {'Expected':<25} {'Predicted':<25} {'Match':<8} BLEU")
        print(f"{'-' * 90}")

        category_correct = 0
        for english, expected_gloss in examples:
            prediction = predict_gloss(
                english,
                bundle,
                device=args.device,
                max_len=32,
                beam_width=args.beam_width,
            )

            predicted_gloss = prediction.predicted_gloss_text
            is_match = (predicted_gloss == expected_gloss)

            if is_match:
                category_correct += 1
                total_correct += 1

            total_tests += 1

            # Compute BLEU
            ref_tokens = expected_gloss.split()
            pred_tokens = predicted_gloss.split()
            bleu_result = compute_bleu(ref_tokens, pred_tokens)
            bleu_score = bleu_result['bleu']
            all_bleu_scores.append(bleu_score)

            # Print result
            marker = "✓" if is_match else "✗"
            if args.show_all or not is_match:
                print(
                    f"{marker} {english:<38} {expected_gloss:<25} {predicted_gloss:<25} "
                    f"{str(is_match):<8} {bleu_score:.4f}"
                )

        accuracy = 100 * category_correct / len(examples)
        print(f"{'-' * 90}")
        print(f"Category accuracy: {category_correct}/{len(examples)} ({accuracy:.1f}%)")

    # Overall statistics
    print(f"\n{'=' * 90}")
    print(f"OVERALL RESULTS")
    print(f"{'=' * 90}")
    print(f"Total correct: {total_correct}/{total_tests} ({100*total_correct/total_tests:.1f}%)")
    print(f"Average BLEU: {sum(all_bleu_scores)/len(all_bleu_scores):.4f}")
    print(f"Min BLEU: {min(all_bleu_scores):.4f}")
    print(f"Max BLEU: {max(all_bleu_scores):.4f}")


if __name__ == "__main__":
    main()
