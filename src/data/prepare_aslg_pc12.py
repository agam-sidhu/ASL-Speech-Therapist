"""Audit and convert the local ASLG-PC12 CSV into repo training format."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_paired_records
from src.data.preprocess_dataset import preprocess_records
from src.nlp.normalize_text import normalize_text


ARTIFACT_PATTERN = re.compile(r"[^A-Za-z0-9'_\-/.,;:?!\s]")
STANDALONE_PUNCTUATION = {".", ",", ";", ":", "!", "?"}
FORMAL_TERMS = {
    "parliament",
    "committee",
    "commission",
    "council",
    "directive",
    "regulation",
    "amendment",
    "motion",
    "resolution",
    "article",
    "budget",
    "report",
    "vote",
    "minister",
    "president",
    "european",
}
CONVERSATIONAL_TERMS = {
    "i",
    "you",
    "me",
    "my",
    "mother",
    "father",
    "brother",
    "sister",
    "friend",
    "today",
    "tomorrow",
    "yesterday",
    "weather",
    "help",
    "hear",
    "understand",
}


def _clean_cell(value: str | None) -> str:
    return " ".join((value or "").replace("\ufeff", "").split())


def _normalize_english(value: str) -> str:
    return normalize_text(_clean_cell(value), remove_fillers=False)["clean_text"]


def _normalize_gloss(value: str) -> str:
    tokens = []
    for token in _clean_cell(value).upper().split():
        if token in STANDALONE_PUNCTUATION:
            continue
        tokens.append(token)
    return " ".join(tokens)


def _tokens(text: str) -> list[str]:
    return [token for token in text.split() if token]


def _lengths(records: list[dict[str, Any]], key: str) -> list[int]:
    return [len(_tokens(str(record[key]))) for record in records]


def _vocab(records: list[dict[str, Any]], key: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(_tokens(str(record[key])))
    return counter


def _summarize_lengths(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        return {"avg": 0, "min": 0, "max": 0}
    return {"avg": round(mean(lengths), 3), "min": min(lengths), "max": max(lengths)}


def _template_key(text: str, keep: int = 4) -> str:
    return " ".join(_tokens(text)[:keep])


def _read_csv(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames or []), rows


def convert_rows(rows: list[dict[str, str]], source_file: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    drop_reasons: Counter[str] = Counter()
    duplicate_pairs = 0
    text_to_glosses: dict[str, set[str]] = defaultdict(set)

    for row_index, row in enumerate(rows):
        english = _normalize_english(row.get("text", ""))
        gloss = _normalize_gloss(row.get("gloss", ""))
        if not english:
            drop_reasons["missing_text"] += 1
            continue
        if not gloss:
            drop_reasons["missing_gloss"] += 1
            continue

        pair_key = (english, gloss)
        text_to_glosses[english].add(gloss)
        if pair_key in seen_pairs:
            duplicate_pairs += 1
            continue
        seen_pairs.add(pair_key)

        converted.append(
            {
                "pair_id": f"aslg_pc12_{len(converted) + 1:06d}",
                "english": english,
                "gloss": gloss,
                "source_kind": "aslg_pc12",
                "source_file": source_file,
                "original_row_index": row_index,
            }
        )

    conflicting_texts = {
        text: sorted(glosses)
        for text, glosses in text_to_glosses.items()
        if len(glosses) > 1
    }
    stats = {
        "rows_seen": len(rows),
        "rows_kept_after_dedup": len(converted),
        "exact_duplicate_pairs_removed": duplicate_pairs,
        "dropped_rows": sum(drop_reasons.values()),
        "drop_reasons": dict(drop_reasons),
        "english_conflict_count": len(conflicting_texts),
        "english_conflict_examples": [
            {"english": text, "glosses": glosses}
            for text, glosses in list(conflicting_texts.items())[:10]
        ],
    }
    return converted, stats


def audit_records(
    *,
    rows: list[dict[str, str]],
    converted: list[dict[str, Any]],
    source_columns: list[str],
    project_dataset: str | None,
) -> dict[str, Any]:
    raw_missing = {
        "text": sum(1 for row in rows if not _clean_cell(row.get("text"))),
        "gloss": sum(1 for row in rows if not _clean_cell(row.get("gloss"))),
    }
    raw_artifact_rows = sum(
        1
        for row in rows
        if ARTIFACT_PATTERN.search(_clean_cell(row.get("text")))
        or ARTIFACT_PATTERN.search(_clean_cell(row.get("gloss")))
    )
    source_lengths = _lengths(converted, "english")
    target_lengths = _lengths(converted, "gloss")
    src_vocab = _vocab(converted, "english")
    tgt_vocab = _vocab(converted, "gloss")
    template_counts = Counter(_template_key(str(item["english"])) for item in converted)
    formal_hits = sum(1 for item in converted if FORMAL_TERMS & set(_tokens(str(item["english"]))))
    conversational_hits = sum(1 for item in converted if CONVERSATIONAL_TERMS & set(_tokens(str(item["english"]))))

    report: dict[str, Any] = {
        "source": "local data/raw/train.csv",
        "columns": source_columns,
        "raw_row_count": len(rows),
        "missing_raw_cells": raw_missing,
        "raw_rows_with_symbol_artifacts": raw_artifact_rows,
        "converted_row_count": len(converted),
        "source_length": _summarize_lengths(source_lengths),
        "target_length": _summarize_lengths(target_lengths),
        "source_vocab_size": len(src_vocab),
        "target_vocab_size": len(tgt_vocab),
        "target_uppercase_rate": round(
            sum(1 for item in converted if str(item["gloss"]) == str(item["gloss"]).upper()) / max(len(converted), 1),
            4,
        ),
        "top_source_tokens": src_vocab.most_common(25),
        "top_target_tokens": tgt_vocab.most_common(25),
        "top_templates": template_counts.most_common(20),
        "formal_domain_row_rate": round(formal_hits / max(len(converted), 1), 4),
        "conversational_term_row_rate": round(conversational_hits / max(len(converted), 1), 4),
        "sample_records": converted[:10],
    }

    if project_dataset:
        project_records = preprocess_records(load_paired_records(project_dataset))
        project_src_vocab = _vocab(project_records, "english")
        project_tgt_vocab = _vocab(project_records, "gloss")
        report["project_comparison"] = {
            "project_dataset": project_dataset,
            "project_row_count": len(project_records),
            "project_source_length": _summarize_lengths(_lengths(project_records, "english")),
            "project_target_length": _summarize_lengths(_lengths(project_records, "gloss")),
            "project_source_vocab_size": len(project_src_vocab),
            "project_target_vocab_size": len(project_tgt_vocab),
            "source_vocab_overlap_with_project": len(set(src_vocab) & set(project_src_vocab)),
            "target_vocab_overlap_with_project": len(set(tgt_vocab) & set(project_tgt_vocab)),
            "project_source_vocab_covered_by_aslg_rate": round(
                len(set(src_vocab) & set(project_src_vocab)) / max(len(project_src_vocab), 1),
                4,
            ),
            "project_target_vocab_covered_by_aslg_rate": round(
                len(set(tgt_vocab) & set(project_tgt_vocab)) / max(len(project_tgt_vocab), 1),
                4,
            ),
        }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and convert the local ASLG-PC12 train.csv.")
    parser.add_argument("--input", default="data/raw/train.csv")
    parser.add_argument("--output", default="data/active/aslg_pc12_pretrain.json")
    parser.add_argument("--report", default="data/reports/data_pipeline_report.json")
    parser.add_argument("--project_dataset", default="data/active/project_finetune_v2_v4_contrastive.json")
    args = parser.parse_args()

    from src.data.build_active_gloss_pipeline import build_active_datasets

    report = build_active_datasets(
        raw_aslg_csv=Path(args.input),
        aslg_output_path=Path(args.output),
        project_output_path=Path(args.project_dataset),
        report_output_path=Path(args.report),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
