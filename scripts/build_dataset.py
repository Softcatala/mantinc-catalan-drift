#!/usr/bin/env python3
"""Validate and summarize the deterministic Catalan Drift dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "monolingual": ROOT / "data/monolingual.yaml",
    "crosslingual_basic": ROOT / "data/crosslingual_basic.yaml",
    "multi_turn": ROOT / "data/multi_turn.yaml",
    "crosslingual_advanced": ROOT / "data/crosslingual_advanced.yaml",
    "rag_context": ROOT / "data/rag_context.yaml",
}
EXPECTED_CATEGORY_COUNTS = {
    "monolingual": 60,
    "crosslingual_basic": 60,
    "multi_turn": 60,
    "crosslingual_advanced": 60,
    "rag_context": 60,
}
VALID_PRESSURE_PATTERNS = {
    "no_pressure",
    "english_or_trilingual",
    "inline_source_es",
    "midconv_es_recency",
    "rag_context",
    "harder_template_es",
    "harder_template_mixed",
    "harder_short_implicit",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or []


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, source in SOURCES.items():
        category_rows = load_rows(source)
        expected_count = EXPECTED_CATEGORY_COUNTS[category]
        if len(category_rows) != expected_count:
            raise ValueError(
                f"{source} must contain {expected_count} rows, found {len(category_rows)}"
            )
        for row in category_rows:
            if row.get("category") != category:
                raise ValueError(
                    f"{row.get('id')} in {source} has category {row.get('category')!r}"
                )
        rows.extend(category_rows)

    validate_rows(rows)
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    expected_categories = Counter(EXPECTED_CATEGORY_COUNTS)
    categories = Counter(str(row.get("category")) for row in rows)
    expected_total = sum(EXPECTED_CATEGORY_COUNTS.values())
    if len(rows) != expected_total or categories != expected_categories:
        raise ValueError(f"unexpected category counts: {dict(categories)}")

    ids = [str(row.get("id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset IDs must be unique")

    required = {
        "id",
        "category",
        "persona",
        "workflow",
        "source_lang",
        "target_lang",
        "prompt",
        "pressure_pattern",
    }
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{row.get('id')} is missing fields: {sorted(missing)}")
        if row["target_lang"] != "ca":
            raise ValueError(f"{row['id']} must target Catalan")
        if row["pressure_pattern"] not in VALID_PRESSURE_PATTERNS:
            raise ValueError(f"{row['id']} has an invalid pressure_pattern")
        if (row["pressure_pattern"] == "no_pressure") != (
            row["category"] == "monolingual"
        ):
            raise ValueError(f"{row['id']} has inconsistent pressure metadata")
        if any(
            field in row
            for field in (
                "labels",
                "version",
                "forbidden_terms",
                "rag_subtype",
                "harder_variant",
            )
        ):
            raise ValueError(f"{row['id']} contains unsupported metadata fields")
        conversation = row.get("conversation") or []
        if any(
            turn.get("role") not in {"user", "assistant"} or not turn.get("content")
            for turn in conversation
        ):
            raise ValueError(f"{row['id']} has an invalid conversation")
        if conversation and conversation[-1]["role"] != "assistant":
            raise ValueError(f"{row['id']} conversation must end with assistant")
        if any(
            left["role"] == right["role"]
            for left, right in zip(conversation, conversation[1:])
        ):
            raise ValueError(f"{row['id']} conversation roles must alternate")


def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    values = [str(row.get(field) or "unknown") for row in rows]
    return dict(sorted(Counter(values).items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=ROOT / "data/prompts.distribution.json"
    )
    args = parser.parse_args()

    rows = build_rows()
    report = {
        "n": len(rows),
        "categories": distribution(rows, "category"),
        "overall": {
            field: distribution(rows, field)
            for field in ("source_lang", "persona", "workflow")
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"sources": list(map(str, SOURCES.values())), "report": str(args.report), "n": len(rows)}))


if __name__ == "__main__":
    main()
