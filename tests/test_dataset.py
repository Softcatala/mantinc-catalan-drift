from __future__ import annotations

import sys
import unittest
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_dataset import (
    EXPECTED_CATEGORY_COUNTS,
    VALID_PRESSURE_PATTERNS,
    build_rows,
)


CATALAN_INSTRUCTION_RE = re.compile(r"\b(?:català|catalana|catalans|catalanes)\b", re.I)
WORD_RE = re.compile(r"[\wÀ-ÿ']+")


def mentions_catalan_instruction(row: dict) -> bool:
    return bool(CATALAN_INSTRUCTION_RE.search(str(row.get("prompt") or "")))


def final_prompt_word_count(row: dict) -> int:
    prompt = str(row.get("prompt") or "")
    prompt = re.sub(r"\ben català\b", "", prompt, flags=re.I)
    prompt = re.sub(r"\b(?:català|catalana|catalans|catalanes)\b", "", prompt, flags=re.I)
    return len(WORD_RE.findall(prompt))


class DatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = build_rows()

    def test_expected_category_counts(self) -> None:
        self.assertEqual(
            Counter(row["category"] for row in self.rows),
            Counter(EXPECTED_CATEGORY_COUNTS),
        )

    def test_pressure_pattern_labels(self) -> None:
        for row in self.rows:
            self.assertIn(row["pressure_pattern"], VALID_PRESSURE_PATTERNS, row["id"])

    def test_rows_do_not_contain_labels(self) -> None:
        self.assertTrue(all("labels" not in row for row in self.rows))

    def test_rows_do_not_contain_forbidden_terms(self) -> None:
        self.assertTrue(all("forbidden_terms" not in row for row in self.rows))

    def test_rag_chunks_do_not_contain_relevance(self) -> None:
        rag_rows = [row for row in self.rows if row["category"] == "rag_context"]
        self.assertTrue(
            all(
                "relevance" not in chunk
                for row in rag_rows
                for chunk in row.get("retrieved_context", [])
            )
        )

    def test_catalan_instruction_policy(self) -> None:
        for row in self.rows:
            with self.subTest(row=row["id"]):
                category = row["category"]
                has_instruction = mentions_catalan_instruction(row)
                if row["pressure_pattern"].startswith("harder_"):
                    self.assertFalse(has_instruction)
                elif category == "rag_context":
                    self.assertTrue(has_instruction)
                elif category == "monolingual":
                    self.assertFalse(has_instruction)
                else:
                    expected = row["source_lang"] != "ca" and final_prompt_word_count(row) < 10
                    self.assertEqual(has_instruction, expected)

    def test_adversarial_pressure_is_distributed_across_task_categories(self) -> None:
        categories = {row["category"] for row in self.rows}
        for category in categories - {"monolingual"}:
            rows = [row for row in self.rows if row["category"] == category]
            self.assertTrue(
                any(row["pressure_pattern"].startswith("harder_") for row in rows),
                category,
            )

    def test_category_matches_item_structure(self) -> None:
        for row in self.rows:
            with self.subTest(row=row["id"]):
                self.assertEqual(
                    bool(row.get("conversation")),
                    row["category"] in {"multi_turn", "crosslingual_advanced"},
                )
                self.assertEqual(
                    bool(row.get("retrieved_context")),
                    row["category"] == "rag_context",
                )
                self.assertEqual(row["source_lang"] == "ca", row["category"] == "monolingual")

    def test_advanced_distribution_preserves_coverage(self) -> None:
        advanced = [row for row in self.rows if row["category"] == "crosslingual_advanced"]
        first_source_languages = Counter(
            row["source_lang"].split("-")[0] for row in advanced
        )
        self.assertEqual(first_source_languages, Counter({"es": 30, "en": 30}))
        self.assertEqual(set(row["persona"] for row in advanced), {"administracio", "pime", "usuari_final"})
        source_workflows = {
            row["workflow"] for row in self.rows if row["category"] == "multi_turn"
        }
        self.assertEqual(set(row["workflow"] for row in advanced), source_workflows)

    def test_build_is_deterministic(self) -> None:
        self.assertEqual(self.rows, build_rows())


if __name__ == "__main__":
    unittest.main()
