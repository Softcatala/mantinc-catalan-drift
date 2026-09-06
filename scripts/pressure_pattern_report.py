#!/usr/bin/env python3
"""Per-pressure-pattern fail rates by model, over outputs/**/samples_*.jsonl.

Reads the consolidated `pressure_pattern` field baked into each dataset item.
"""

from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ORDER = [
    "no_pressure",
    "english_or_trilingual",
    "inline_source_es",
    "midconv_es_recency",
    "rag_context",
    "harder_template_es",
    "harder_template_mixed",
    "harder_short_implicit",
]
COMPACT_LABELS = {
    "no_pressure": "control",
    "english_or_trilingual": "en/tri",
    "inline_source_es": "inline-es",
    "midconv_es_recency": "midconv-es",
    "rag_context": "rag",
    "harder_template_es": "hard-tmpl",
    "harder_template_mixed": "hard-mix",
    "harder_short_implicit": "short-impl",
}

items = {}
for path in sorted(ROOT.glob("data/*.yaml")):
    for r in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
        items[r["id"]] = r.get("pressure_pattern", "unknown")

results = defaultdict(dict)
for p in sorted(ROOT.glob("outputs/*/lm_eval/**/samples_*.jsonl")):
    model = p.relative_to(ROOT).parts[1]
    for line in open(p, encoding="utf-8"):
        s = json.loads(line)
        did = s.get("doc", {}).get("id")
        if did:
            results[model][did] = float(s.get("drift_pass", 0)) == 1.0

models = sorted(model for model, samples in results.items() if items.keys() & samples.keys())
agg = defaultdict(lambda: [0, 0])
totals = defaultdict(int)
for did, pat in items.items():
    totals[pat] += 1
    for m in models:
        if did in results[m]:
            agg[(pat, m)][0] += int(results[m][did])
            agg[(pat, m)][1] += 1


def short(m):
    for suf in ("-q4_k_m", "-instruct-2512"):
        m = m.replace(suf, "")
    return m.replace("gemma-4-", "gemma-").replace("gemini-3.7-", "gemini-")


def fail_rate(pattern, model):
    passed, evaluated = agg[(pattern, model)]
    return 100 * (evaluated - passed) / evaluated if evaluated else None


def cell(value):
    return f"{value:5.1f}%" if value is not None else "    -"


def print_table(rows):
    widths = [max(len(c) for c in column) for column in zip(*rows)]
    for i, row in enumerate(rows):
        print("  ".join([row[0].ljust(widths[0])] + [row[j].rjust(widths[j]) for j in range(1, len(row))]))
        if i == 0:
            print("  ".join("─" * width for width in widths))


def wide_rows(patterns):
    rows = [["pattern", "n"] + [short(model) for model in models] + ["avg"]]
    for pattern in patterns:
        rates = [fail_rate(pattern, model) for model in models]
        present = [rate for rate in rates if rate is not None]
        rows.append(
            [pattern, str(totals[pattern])]
            + [cell(rate) for rate in rates]
            + [cell(sum(present) / len(present) if present else None)]
        )
    return rows


def compact_rows(patterns):
    rows = [["model / run"] + [COMPACT_LABELS[pattern] for pattern in patterns] + ["avg"]]
    rows.append(["dataset n"] + [str(totals[pattern]) for pattern in patterns] + [""])
    all_rates = []
    for model in models:
        rates = [fail_rate(pattern, model) for pattern in patterns]
        present = [rate for rate in rates if rate is not None]
        all_rates.extend(present)
        rows.append(
            [short(model)]
            + [cell(rate) for rate in rates]
            + [cell(sum(present) / len(present) if present else None)]
        )
    pattern_averages = []
    for pattern in patterns:
        present = [fail_rate(pattern, model) for model in models]
        present = [rate for rate in present if rate is not None]
        pattern_averages.append(sum(present) / len(present) if present else None)
    overall = sum(all_rates) / len(all_rates) if all_rates else None
    rows.append(["model avg"] + [cell(rate) for rate in pattern_averages] + [cell(overall)])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wide",
        action="store_true",
        help="put models in columns (the original, spreadsheet-friendly layout)",
    )
    args = parser.parse_args()
    patterns = [pattern for pattern in ORDER if totals.get(pattern)]
    print_table(wide_rows(patterns) if args.wide else compact_rows(patterns))
    print()
    print("Cells are % fail ratio among evaluated items; '-' means no results.")
    if not args.wide:
        print("Pattern labels: " + ", ".join(f"{COMPACT_LABELS[p]}={p}" for p in patterns))


if __name__ == "__main__":
    main()
