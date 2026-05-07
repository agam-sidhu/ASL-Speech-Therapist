"""Basic ASR metrics without external dependencies."""

from __future__ import annotations


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Compute Levenshtein edit distance."""
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    """Compute WER from tokenized word sequences."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(reference, hypothesis) / len(reference)


def char_error_rate(reference: str, hypothesis: str) -> float:
    """Compute CER from normalized strings."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(list(reference), list(hypothesis)) / len(reference)
