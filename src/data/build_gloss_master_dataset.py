"""Compatibility wrapper for the active gloss data pipeline."""

from __future__ import annotations

import json

from src.data.build_active_gloss_pipeline import build_active_datasets


def main() -> None:
    report = build_active_datasets()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
