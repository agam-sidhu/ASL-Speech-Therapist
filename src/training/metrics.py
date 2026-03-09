"""Evaluation metrics for English->ASL gloss translation quality."""

from __future__ import annotations

import math
from collections import Counter


def _ngrams(tokens: list[str], n: int) -> Counter:
    """Extract n-gram counts from a token list."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def compute_bleu(
    reference: list[str],
    hypothesis: list[str],
    max_n: int = 4,
    smooth: bool = True,
) -> dict[str, float]:
    """Compute BLEU score between a reference and hypothesis gloss sequence.

    Uses smoothed BLEU (add-1 smoothing) by default to handle short sequences,
    which is important for ASL gloss where outputs are typically short.

    Args:
        reference: List of reference gloss tokens (ground truth).
        hypothesis: List of predicted gloss tokens (model output).
        max_n: Maximum n-gram order (default 4 for BLEU-4).
        smooth: Whether to apply add-1 smoothing for zero counts.

    Returns:
        Dictionary with 'bleu', 'brevity_penalty', and per-n-gram precisions.
    """
    if not hypothesis:
        return {"bleu": 0.0, "brevity_penalty": 0.0, **{f"p{i}": 0.0 for i in range(1, max_n + 1)}}

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        ref_ngrams = _ngrams(reference, n)
        hyp_ngrams = _ngrams(hypothesis, n)

        clipped = 0
        total = 0
        for ngram, count in hyp_ngrams.items():
            clipped += min(count, ref_ngrams.get(ngram, 0))
            total += count

        if total == 0:
            if smooth:
                precisions.append(1.0 / (len(hypothesis) + 1))
            else:
                precisions.append(0.0)
        else:
            if smooth and clipped == 0:
                precisions.append(1.0 / (total + 1))
            else:
                precisions.append(clipped / total)

    # Brevity penalty
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    if hyp_len >= ref_len:
        bp = 1.0
    elif hyp_len == 0:
        bp = 0.0
    else:
        bp = math.exp(1.0 - ref_len / hyp_len)

    # Geometric mean of precisions
    log_precision = 0.0
    for p in precisions:
        if p <= 0:
            log_precision = float("-inf")
            break
        log_precision += math.log(p)
    log_precision /= max_n

    bleu = bp * math.exp(log_precision) if log_precision > float("-inf") else 0.0

    result = {"bleu": bleu, "brevity_penalty": bp}
    for i, p in enumerate(precisions, 1):
        result[f"p{i}"] = p
    return result


def corpus_bleu(
    references: list[list[str]],
    hypotheses: list[list[str]],
    max_n: int = 4,
) -> dict[str, float]:
    """Compute corpus-level BLEU score over multiple reference/hypothesis pairs.

    Args:
        references: List of reference gloss token lists.
        hypotheses: List of hypothesis gloss token lists.
        max_n: Maximum n-gram order.

    Returns:
        Corpus-level BLEU dictionary.
    """
    assert len(references) == len(hypotheses), "Mismatched number of references and hypotheses."

    total_clipped = [0] * max_n
    total_count = [0] * max_n
    total_ref_len = 0
    total_hyp_len = 0

    for ref, hyp in zip(references, hypotheses):
        total_ref_len += len(ref)
        total_hyp_len += len(hyp)

        for n in range(1, max_n + 1):
            ref_ngrams = _ngrams(ref, n)
            hyp_ngrams = _ngrams(hyp, n)
            for ngram, count in hyp_ngrams.items():
                total_clipped[n - 1] += min(count, ref_ngrams.get(ngram, 0))
                total_count[n - 1] += count

    precisions = []
    for n in range(max_n):
        if total_count[n] == 0:
            precisions.append(0.0)
        else:
            precisions.append(total_clipped[n] / total_count[n])

    if total_hyp_len == 0:
        return {"corpus_bleu": 0.0, "brevity_penalty": 0.0}

    bp = 1.0 if total_hyp_len >= total_ref_len else math.exp(1.0 - total_ref_len / total_hyp_len)

    log_precision = 0.0
    for p in precisions:
        if p <= 0:
            return {"corpus_bleu": 0.0, "brevity_penalty": bp, **{f"p{i+1}": precisions[i] for i in range(max_n)}}
        log_precision += math.log(p)
    log_precision /= max_n

    bleu = bp * math.exp(log_precision)
    result = {"corpus_bleu": bleu, "brevity_penalty": bp}
    for i, p in enumerate(precisions, 1):
        result[f"p{i}"] = p
    return result
