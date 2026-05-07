"""Deterministic dataset splitting helpers for English->ASL experiments."""

from __future__ import annotations

import random


def split_records(
    records: list[dict[str, str]],
    *,
    val_split: float = 0.15,
    test_split: float = 0.0,
    seed: int = 42,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Split records into train/val/test with a reproducible shuffle."""
    if not records:
        raise ValueError("Cannot split an empty dataset.")
    if val_split < 0 or test_split < 0:
        raise ValueError("val_split and test_split must be non-negative.")
    if val_split + test_split >= 1:
        raise ValueError("val_split + test_split must be < 1.")

    shuffled = records[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)
    test_count = int(total * test_split)
    val_count = int(total * val_split)

    if test_split > 0 and test_count == 0 and total >= 3:
        test_count = 1
    if val_split > 0 and val_count == 0 and total - test_count >= 2:
        val_count = 1

    if test_count + val_count >= total:
        raise ValueError("Split sizes leave no training examples. Reduce val/test split ratios.")

    test_records = shuffled[:test_count]
    val_records = shuffled[test_count : test_count + val_count]
    train_records = shuffled[test_count + val_count :]

    if not train_records:
        raise ValueError("Training split is empty after applying val/test split.")

    return train_records, val_records, test_records


def select_split(
    records: list[dict[str, str]],
    *,
    split: str,
    val_split: float,
    test_split: float,
    seed: int,
) -> list[dict[str, str]]:
    """Select a named split using the same deterministic partitioning rules."""
    split = split.lower()
    if split == "all":
        return records

    train_records, val_records, test_records = split_records(
        records,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )
    if split == "train":
        return train_records
    if split == "val":
        return val_records
    if split == "test":
        return test_records
    raise ValueError(f"Unsupported split '{split}'. Use all/train/val/test.")
