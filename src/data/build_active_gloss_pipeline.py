"""Build the active gloss datasets for the demo training pipeline."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from src.data.build_contrastive_gloss_pairs import CONTRASTIVE_GROUPS
from src.data.build_gloss_pair_datasets import GENERATED_PAIRS
from src.nlp.normalize_text import normalize_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RAW_ASLG_CSV = PROJECT_ROOT / "data" / "raw" / "train.csv"
DEFAULT_RAW_CONVERSATIONAL = PROJECT_ROOT / "data" / "raw" / "asl_gloss_conversational.json"
DEFAULT_RAW_V2 = PROJECT_ROOT / "data" / "raw" / "asl_gloss_pairs_v2.json"
DEFAULT_RAW_V4 = PROJECT_ROOT / "data" / "raw" / "asl_gloss_pairs_v4.json"
DEFAULT_ASLG_OUTPUT = PROJECT_ROOT / "data" / "active" / "aslg_pc12_pretrain.json"
DEFAULT_PROJECT_OUTPUT = PROJECT_ROOT / "data" / "active" / "project_finetune_v2_v4_contrastive.json"
DEFAULT_REPORT_OUTPUT = PROJECT_ROOT / "data" / "reports" / "data_pipeline_report.json"

ARTIFACT_PATTERN = re.compile(r"[^A-Za-z0-9'_\-/.,;:?!\s]")
STANDALONE_PUNCTUATION = {".", ",", ";", ":", "!", "?"}


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def clean_cell(value: str | None) -> str:
    return " ".join((value or "").replace("\ufeff", "").split())


def normalize_english(value: str) -> str:
    return normalize_text(clean_cell(value), remove_fillers=False)["clean_text"]


def normalize_gloss(value: str) -> str:
    tokens = []
    for token in clean_cell(value).upper().split():
        if token in STANDALONE_PUNCTUATION:
            continue
        tokens.append(token)
    return " ".join(tokens)


def ensure_required_raw_files(paths: list[Path]) -> None:
    missing = [rel(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required raw files are missing: {missing}")


def validate_no_active_archive_or_review_read(paths: list[Path]) -> None:
    blocked_parts = {("data", "archive"), ("data", "review")}
    for path in paths:
        parts = path.relative_to(PROJECT_ROOT).parts if path.is_absolute() else path.parts
        for first, second in blocked_parts:
            if len(parts) >= 2 and parts[0] == first and parts[1] == second:
                raise ValueError(f"Active pipeline cannot read from {first}/{second}: {path}")


def validate_no_v3_references(records: list[dict[str, Any]], dataset_label: str) -> None:
    hits: list[str] = []
    for index, record in enumerate(records):
        source_values = []
        for key in ("source_file", "source_files"):
            value = record.get(key)
            if isinstance(value, list):
                source_values.extend(str(item) for item in value)
            elif value is not None:
                source_values.append(str(value))
        if any("v3" in item.lower() for item in source_values):
            hits.append(f"{dataset_label}[{index}]")
    if hits:
        raise ValueError(f"v3 source references found in active dataset: {hits[:10]}")


def validate_no_conflicting_english(records: list[dict[str, Any]], dataset_label: str) -> int:
    english_to_glosses: dict[str, set[str]] = defaultdict(set)
    for record in records:
        english_to_glosses[normalize_english(str(record["english"]))].add(normalize_gloss(str(record["gloss"])))
    conflicts = {english: glosses for english, glosses in english_to_glosses.items() if len(glosses) > 1}
    if conflicts:
        examples = [
            {"english": english, "glosses": sorted(glosses)}
            for english, glosses in list(conflicts.items())[:10]
        ]
        raise ValueError(f"{dataset_label} has conflicting English-to-gloss mappings: {examples}")
    return len(conflicts)


def load_raw_project_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list: {rel(path)}")
    records: list[dict[str, Any]] = []
    duplicate_pairs = 0
    seen_pairs: set[tuple[str, str]] = set()
    for item in payload:
        english = normalize_english(str(item["english"]))
        gloss = normalize_gloss(str(item["gloss"]))
        if not english or not gloss:
            continue
        pair = (english, gloss)
        if pair in seen_pairs:
            duplicate_pairs += 1
            continue
        seen_pairs.add(pair)
        records.append(
            {
                "english": english,
                "gloss": gloss,
                "source_kind": "project_v2_v4_conversational",
                "source_files": [rel(path)],
            }
        )
    return records, {"raw_count": len(payload), "duplicates_removed": duplicate_pairs}


def build_generated_records(existing_english: set[str], existing_pairs: set[tuple[str, str]]) -> list[dict[str, Any]]:
    generated: list[dict[str, Any]] = []
    seen_english: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for category, pairs in GENERATED_PAIRS.items():
        for english_text, gloss_text in pairs:
            english = normalize_english(english_text)
            gloss = normalize_gloss(gloss_text)
            pair = (english, gloss)
            if english in existing_english or pair in existing_pairs:
                raise ValueError(f"Generated augmentation overlaps project base: {(english, gloss)}")
            if english in seen_english or pair in seen_pairs:
                raise ValueError(f"Generated augmentation duplicates internally: {(english, gloss)}")
            generated.append(
                {
                    "english": english,
                    "gloss": gloss,
                    "source_kind": "generated_augmentation",
                    "source_files": ["in_memory:GENERATED_PAIRS"],
                    "manual_category": category,
                    "notes": "Synthetic grammar-focused augmentation generated by the active data pipeline.",
                }
            )
            seen_english.add(english)
            seen_pairs.add(pair)
    if len(generated) != 150:
        raise ValueError(f"Expected 150 generated augmentation records, found {len(generated)}")
    return generated


def build_contrastive_records(existing_english: set[str], existing_pairs: set[tuple[str, str]]) -> list[dict[str, Any]]:
    contrastive: list[dict[str, Any]] = []
    seen_english: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    for group in CONTRASTIVE_GROUPS:
        examples = list(group["examples"])
        if len(examples) != 5:
            raise ValueError(f"Contrast group {group['contrast_group']} must contain 5 examples.")
        for english_text, gloss_text in examples:
            english = normalize_english(english_text)
            gloss = normalize_gloss(gloss_text)
            pair = (english, gloss)
            if english in existing_english or pair in existing_pairs:
                raise ValueError(f"Contrastive augmentation overlaps earlier project data: {(english, gloss)}")
            if english in seen_english or pair in seen_pairs:
                raise ValueError(f"Contrastive augmentation duplicates internally: {(english, gloss)}")
            contrastive.append(
                {
                    "english": english,
                    "gloss": gloss,
                    "source_kind": "contrastive_generated",
                    "source_files": ["in_memory:CONTRASTIVE_GROUPS"],
                    "manual_category": str(group["manual_category"]),
                    "contrast_group": str(group["contrast_group"]),
                    "contrast_axis": str(group["contrast_axis"]),
                    "notes": "Synthetic contrastive augmentation generated by the active data pipeline.",
                }
            )
            seen_english.add(english)
            seen_pairs.add(pair)
    if len(contrastive) != 200:
        raise ValueError(f"Expected 200 contrastive records, found {len(contrastive)}")
    return contrastive


def build_project_finetune_dataset(raw_conversational_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_records, base_stats = load_raw_project_records(raw_conversational_path)
    existing_pairs = {(item["english"], item["gloss"]) for item in base_records}
    existing_english = {str(item["english"]) for item in base_records}

    generated = build_generated_records(existing_english, existing_pairs)
    existing_pairs.update((item["english"], item["gloss"]) for item in generated)
    existing_english.update(str(item["english"]) for item in generated)

    contrastive = build_contrastive_records(existing_english, existing_pairs)

    final_records: list[dict[str, Any]] = []
    duplicate_pairs_removed = 0
    seen_pairs: set[tuple[str, str]] = set()
    for item in base_records + generated + contrastive:
        pair = (str(item["english"]), str(item["gloss"]))
        if pair in seen_pairs:
            duplicate_pairs_removed += 1
            continue
        seen_pairs.add(pair)
        record = dict(item)
        record["pair_id"] = f"project_ft_{len(final_records) + 1:04d}"
        final_records.append(record)

    validate_no_conflicting_english(final_records, "project fine-tune")
    validate_no_v3_references(final_records, "project fine-tune")
    if not final_records:
        raise ValueError("Project fine-tune dataset output count is zero.")

    report = {
        "raw_project_pair_count": base_stats["raw_count"],
        "project_base_pair_count": len(base_records),
        "generated_augmentation_count": len(generated),
        "contrastive_pair_count": len(contrastive),
        "final_project_finetune_pair_count": len(final_records),
        "project_duplicates_removed": base_stats["duplicates_removed"] + duplicate_pairs_removed,
        "project_conflicts_removed": 0,
        "source_kind_counts": dict(Counter(str(item["source_kind"]) for item in final_records)),
    }
    return final_records, report


def read_aslg_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames or []), rows


def build_aslg_pretrain_dataset(csv_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    columns, rows = read_aslg_rows(csv_path)
    converted: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    drop_reasons: Counter[str] = Counter()
    duplicate_pairs_removed = 0
    text_to_glosses: dict[str, set[str]] = defaultdict(set)
    artifact_rows = 0

    for row_index, row in enumerate(rows):
        raw_text = clean_cell(row.get("text"))
        raw_gloss = clean_cell(row.get("gloss"))
        if ARTIFACT_PATTERN.search(raw_text) or ARTIFACT_PATTERN.search(raw_gloss):
            artifact_rows += 1
        english = normalize_english(raw_text)
        gloss = normalize_gloss(raw_gloss)
        if not english:
            drop_reasons["missing_text"] += 1
            continue
        if not gloss:
            drop_reasons["missing_gloss"] += 1
            continue
        pair = (english, gloss)
        text_to_glosses[english].add(gloss)
        if pair in seen_pairs:
            duplicate_pairs_removed += 1
            continue
        seen_pairs.add(pair)
        converted.append(
            {
                "pair_id": f"aslg_pc12_{len(converted) + 1:06d}",
                "english": english,
                "gloss": gloss,
                "source_kind": "aslg_pc12",
                "source_file": rel(csv_path),
                "original_row_index": row_index,
            }
        )

    validate_no_v3_references(converted, "ASLG-PC12 pretrain")
    if not converted:
        raise ValueError("ASLG-PC12 pretrain dataset output count is zero.")

    conflict_count = sum(1 for glosses in text_to_glosses.values() if len(glosses) > 1)
    lengths = [len(str(item["gloss"]).split()) for item in converted]
    report = {
        "raw_aslg_row_count": len(rows),
        "aslg_columns": columns,
        "cleaned_aslg_pair_count": len(converted),
        "aslg_duplicates_removed": duplicate_pairs_removed,
        "aslg_dropped_rows": sum(drop_reasons.values()),
        "aslg_drop_reasons": dict(drop_reasons),
        "aslg_conflicting_english_count": conflict_count,
        "aslg_rows_with_symbol_artifacts": artifact_rows,
        "aslg_target_length": {
            "avg": round(mean(lengths), 3) if lengths else 0,
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
    }
    return converted, report


def build_active_datasets(
    *,
    raw_aslg_csv: Path = DEFAULT_RAW_ASLG_CSV,
    raw_conversational_path: Path = DEFAULT_RAW_CONVERSATIONAL,
    raw_v2_path: Path = DEFAULT_RAW_V2,
    raw_v4_path: Path = DEFAULT_RAW_V4,
    aslg_output_path: Path = DEFAULT_ASLG_OUTPUT,
    project_output_path: Path = DEFAULT_PROJECT_OUTPUT,
    report_output_path: Path = DEFAULT_REPORT_OUTPUT,
) -> dict[str, Any]:
    raw_paths = [raw_aslg_csv, raw_conversational_path, raw_v2_path, raw_v4_path]
    ensure_required_raw_files(raw_paths)
    validate_no_active_archive_or_review_read(raw_paths)

    aslg_records, aslg_report = build_aslg_pretrain_dataset(raw_aslg_csv)
    project_records, project_report = build_project_finetune_dataset(raw_conversational_path)

    validate_no_active_archive_or_review_read([aslg_output_path, project_output_path, report_output_path])
    validate_no_v3_references(aslg_records, "ASLG-PC12 pretrain")
    validate_no_v3_references(project_records, "project fine-tune")

    write_json(aslg_output_path, aslg_records)
    write_json(project_output_path, project_records)

    report = {
        "status": "ok",
        "v3_detected": False,
        "raw_inputs": {
            "aslg_pc12_csv": rel(raw_aslg_csv),
            "project_conversational": rel(raw_conversational_path),
            "v2_provenance": rel(raw_v2_path),
            "v4_provenance": rel(raw_v4_path),
        },
        "active_outputs": {
            "aslg_pc12_pretrain": rel(aslg_output_path),
            "project_finetune_v2_v4_contrastive": rel(project_output_path),
            "report": rel(report_output_path),
        },
        **aslg_report,
        **project_report,
        "total_duplicates_removed": aslg_report["aslg_duplicates_removed"] + project_report["project_duplicates_removed"],
        "total_conflicts_removed": project_report["project_conflicts_removed"],
    }
    write_json(report_output_path, report)
    return report


def main() -> None:
    report = build_active_datasets()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
