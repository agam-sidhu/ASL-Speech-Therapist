"""Translation-quality analysis focused on ASL-style gloss behavior."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from src.training.metrics import compute_bleu
from src.utils.config import GLOSS_STOPWORDS

WORD_PATTERN = re.compile(r"[A-Za-z']+")
UPPER_STOPWORDS = {token.upper() for token in GLOSS_STOPWORDS}
TIME_TOKENS = {
    "TODAY",
    "TOMORROW",
    "YESTERDAY",
    "NOW",
    "LATER",
    "TIME",
    "EVERY-DAY",
    "MORNING",
    "AFTERNOON",
    "NIGHT",
}
WH_TOKENS = {"WHAT", "WHERE", "WHEN", "WHY", "HOW", "HOW-MUCH", "HOW-LONG", "HOW-MANY"}
YES_NO_PROMPT_TOKENS = {"DO", "ARE", "IS", "CAN", "WILL", "WOULD", "DID", "HAVE", "HAS"}
PRONOUN_TOKENS = {"I", "YOU", "HE", "SHE", "WE", "THEY", "IT", "MY", "YOUR", "OUR", "THEIR"}


def english_tokens_for_analysis(text: str) -> list[str]:
    """Normalize English text into uppercase lexical tokens for comparison."""
    return [token.upper() for token in WORD_PATTERN.findall((text or "").lower())]


def aligned_token_accuracy(reference: list[str], hypothesis: list[str]) -> float:
    """Simple aligned token accuracy over the longer sequence length."""
    denom = max(len(reference), len(hypothesis), 1)
    correct = 0
    for ref_token, hyp_token in zip(reference, hypothesis):
        if ref_token == hyp_token:
            correct += 1
    return correct / denom


def token_overlap_f1(reference: list[str], hypothesis: list[str]) -> float:
    """Bag-of-tokens F1 to complement exact match and BLEU."""
    if not reference and not hypothesis:
        return 1.0
    if not reference or not hypothesis:
        return 0.0

    matched = 0
    remaining = list(hypothesis)
    for token in reference:
        if token in remaining:
            remaining.remove(token)
            matched += 1

    precision = matched / len(hypothesis)
    recall = matched / len(reference)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _shared_order(sequence: list[str], universe: list[str]) -> list[str]:
    return [token for token in sequence if token in universe]


def _function_words(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token in UPPER_STOPWORDS]


def requires_reordering(english_tokens: list[str], reference_tokens: list[str]) -> bool:
    """Heuristic: reference order differs from English order over shared tokens."""
    shared_ref = _shared_order(reference_tokens, english_tokens)
    shared_eng = _shared_order(english_tokens, shared_ref)
    return len(shared_ref) >= 2 and shared_ref != shared_eng


def follows_reference_order(
    english_tokens: list[str],
    reference_tokens: list[str],
    hypothesis_tokens: list[str],
) -> bool:
    """Check whether the hypothesis follows reference order over shared content tokens."""
    shared_ref = _shared_order(reference_tokens, english_tokens)
    common = [token for token in shared_ref if token in hypothesis_tokens]
    if len(common) < 2:
        return False
    ref_common = _shared_order(shared_ref, common)
    hyp_common = _shared_order(hypothesis_tokens, common)
    return hyp_common == ref_common


def copies_english_order(english_tokens: list[str], hypothesis_tokens: list[str]) -> bool:
    """Detect shallow copying of English token order."""
    if not hypothesis_tokens:
        return False
    shared_eng = _shared_order(english_tokens, hypothesis_tokens)
    shared_hyp = _shared_order(hypothesis_tokens, english_tokens)
    return len(shared_hyp) >= 2 and shared_hyp == shared_eng


def reorder_strength(english_tokens: list[str], reference_tokens: list[str]) -> str:
    """Classify how strongly the reference departs from English token order."""
    shared_ref = _shared_order(reference_tokens, english_tokens)
    shared_eng = _shared_order(english_tokens, shared_ref)
    if len(shared_ref) < 2 or shared_ref == shared_eng:
        return "none"

    if reference_tokens and reference_tokens[0] in TIME_TOKENS:
        return "strong"
    if reference_tokens and reference_tokens[-1] in WH_TOKENS:
        return "strong"
    if len(shared_ref) >= 3:
        return "strong"
    return "mild"


def classify_reference_categories(english_text: str, reference_gloss: str) -> list[str]:
    """Assign grammar-oriented categories from the reference pair."""
    english_tokens = english_tokens_for_analysis(english_text)
    reference_tokens = [token.upper() for token in reference_gloss.split()]

    categories: list[str] = []
    strength = reorder_strength(english_tokens, reference_tokens)

    if english_tokens == reference_tokens:
        categories.append("exact_copy")
    elif strength == "mild":
        categories.append("mild_reorder")
    elif strength == "strong":
        categories.append("strong_reorder")
    elif Counter(reference_tokens).items() and set(reference_tokens).issubset(set(english_tokens)):
        categories.append("deletion_or_subset")

    if _function_words(english_tokens) and not _function_words(reference_tokens):
        categories.append("function_word_drop")
    elif _function_words(reference_tokens):
        categories.append("function_word_retained")

    if reference_tokens and reference_tokens[-1] in WH_TOKENS:
        categories.append("wh_question")
    if "NOT" in reference_tokens or "NEVER" in reference_tokens:
        categories.append("negation")
    if reference_tokens and reference_tokens[0] in TIME_TOKENS:
        categories.append("time_fronting")
    if (
        english_tokens
        and english_tokens[0] in YES_NO_PROMPT_TOKENS
        and not (reference_tokens and reference_tokens[-1] in WH_TOKENS)
    ):
        categories.append("yes_no_question")
    if reference_tokens and english_tokens and reference_tokens[0] == english_tokens[0] and reference_tokens[0] in {"MY", "YOUR", "THIS", "THAT"}:
        categories.append("topic_comment_like")
    if any(token.endswith("-TO") or token.endswith("-BACK") for token in reference_tokens):
        categories.append("lexicalized_gloss")

    return categories


def coarse_gloss_template(reference_gloss: str) -> str:
    """Collapse gloss tokens into a grammar-oriented coarse template."""
    template: list[str] = []
    for token in reference_gloss.upper().split():
        if token in TIME_TOKENS or token in WH_TOKENS or token in {"NOT", "NEVER"}:
            template.append(token)
        else:
            template.append("X")
    return " ".join(template)


def explain_case_differences(reference_tokens: list[str], predicted_tokens: list[str]) -> list[str]:
    """Produce short, interpretable error notes."""
    notes: list[str] = []
    missing = [token for token in reference_tokens if token not in predicted_tokens]
    extra = [token for token in predicted_tokens if token not in reference_tokens]
    if missing:
        notes.append(f"missing:{','.join(missing[:3])}")
    if extra:
        notes.append(f"extra:{','.join(extra[:3])}")
    return notes


def analyze_translation_case(
    english_text: str,
    reference_gloss: str,
    predicted_gloss_tokens: list[str],
) -> dict[str, Any]:
    """Compute example-level translation diagnostics."""
    english_tokens = english_tokens_for_analysis(english_text)
    reference_tokens = [token.upper() for token in reference_gloss.split()]
    predicted_tokens = [token.upper() for token in predicted_gloss_tokens]

    reorder_required = requires_reordering(english_tokens, reference_tokens)
    follows_ref_order = follows_reference_order(english_tokens, reference_tokens, predicted_tokens)
    english_copy = copies_english_order(english_tokens, predicted_tokens)
    retained_function_words = [token for token in predicted_tokens if token in UPPER_STOPWORDS]
    bleu = compute_bleu(reference_tokens, predicted_tokens)["bleu"]
    reference_categories = classify_reference_categories(english_text, reference_gloss)

    notes: list[str] = []
    if reorder_required and not follows_ref_order:
        notes.append("missed_reference_reordering")
    if english_copy and reorder_required:
        notes.append("copied_english_order")
    if retained_function_words:
        notes.append("retained_function_words")
    if not predicted_tokens:
        notes.append("empty_prediction")
    notes.extend(explain_case_differences(reference_tokens, predicted_tokens))

    return {
        "english_tokens": english_tokens,
        "reference_tokens": reference_tokens,
        "predicted_tokens": predicted_tokens,
        "exact_match": predicted_tokens == reference_tokens,
        "bleu": bleu,
        "aligned_token_accuracy": aligned_token_accuracy(reference_tokens, predicted_tokens),
        "token_overlap_f1": token_overlap_f1(reference_tokens, predicted_tokens),
        "reorder_required": reorder_required,
        "follows_reference_order": follows_ref_order,
        "copies_english_order": english_copy,
        "retained_function_words": retained_function_words,
        "well_formed": bool(predicted_tokens) and all(token == token.upper() for token in predicted_tokens),
        "reorder_strength": reorder_strength(english_tokens, reference_tokens),
        "reference_categories": reference_categories,
        "reference_template": coarse_gloss_template(reference_gloss),
        "notes": notes,
    }
