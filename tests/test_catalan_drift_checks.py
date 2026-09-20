from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lm_eval_tasks.catalan_drift.utils import (
    _validate_language_id_model,
    catalan_token_ratio,
    lcb_line_language_result,
    process_results,
)


def fake_fasttext(text: str) -> tuple[str, float]:
    lowered = text.casefold()
    if "pedido" in lowered or "retrasado" in lowered:
        return "es", 0.99
    return "ca", 0.99


def medium_confidence_false_positive(text: str) -> tuple[str, float]:
    if "especialista no estava disponible" in text.casefold():
        return "es", 0.648
    return "ca", 0.99


def one_bullet_false_positive(text: str) -> tuple[str, float]:
    if "cita confirmada" in text.casefold():
        return "es", 0.795
    return "ca", 0.99


class LanguageIdModelValidationTest(unittest.TestCase):
    def test_missing_model_fails_before_evaluation(self) -> None:
        missing = "/tmp/mantinc-missing-lid.176.ftz"
        with mock.patch.dict(os.environ, {"LANGUAGE_ID_MODEL": missing}):
            with self.assertRaisesRegex(RuntimeError, "fastText language-ID model not found"):
                _validate_language_id_model()

    def test_environment_model_is_accepted(self) -> None:
        with mock.patch.dict(os.environ, {"LANGUAGE_ID_MODEL": __file__}):
            self.assertEqual(_validate_language_id_model(), Path(__file__))


class CatalanDriftChecksTest(unittest.TestCase):
    def test_segment_language_ratio_can_fail_response(self) -> None:
        response = (
            "Aquest text català manté una resposta completa amb informació clara "
            "i ordenada per al client.\n\n"
            "El pedido queda retrasado sin fecha nueva."
        )

        with mock.patch(
            "lm_eval_tasks.catalan_drift.utils._predict_fasttext",
            side_effect=fake_fasttext,
        ):
            result = process_results({"target_lang": "ca", "category": "test"}, [response])

        self.assertEqual(result["language_fail"], 1.0)
        self.assertEqual(result["drift_pass"], 0.0)
        self.assertEqual(result["catalan_token_ratio"], (15, 22))
        self.assertAlmostEqual(catalan_token_ratio([result["catalan_token_ratio"]]), 15 / 22)

    def test_medium_confidence_romance_confusion_does_not_fail(self) -> None:
        response = (
            "Aquesta reclamació està redactada en català i manté una estructura formal "
            "amb cronologia, impacte i petició concreta.\n\n"
            "En arribar al centre, se'm va informar que l'especialista no estava "
            "disponible i que la visita no es podia dur a terme."
        )

        with mock.patch(
            "lm_eval_tasks.catalan_drift.utils._predict_fasttext",
            side_effect=medium_confidence_false_positive,
        ):
            result = process_results({"target_lang": "ca", "category": "test"}, [response])

        self.assertEqual(result["language_fail"], 0.0)
        self.assertEqual(result["drift_pass"], 1.0)

    def test_one_misclassified_catalan_bullet_does_not_fail_response(self) -> None:
        response = (
            "El punt de vacunació habitual es trasllada temporalment a un centre "
            "alternatiu. Les cites ja confirmades es mantenen, però s'atendran a la "
            "nova ubicació, que disposarà d'horari ampliat i personal de suport. "
            "Hi ha opcions de transport públic properes i personal disponible per "
            "orientar les persones usuàries durant el canvi temporal.\n\n"
            "- Comproveu l'adreça del centre alternatiu abans de desplaçar-vos-hi.\n"
            "- Arribeu amb antelació i porteu la documentació necessària.\n"
            "- Si teniu una cita confirmada, no cal que en demaneu una de nova.\n"
            "- Tingueu en compte possibles retards durant el període d'adaptació.\n"
            "- Si necessiteu ajuda per arribar-hi, contacteu amb el servei d'atenció."
        )

        with mock.patch(
            "lm_eval_tasks.catalan_drift.utils._predict_fasttext",
            side_effect=one_bullet_false_positive,
        ):
            result = process_results({"target_lang": "ca", "category": "test"}, [response])

        self.assertEqual(result["language_fail"], 0.0)
        self.assertEqual(result["drift_pass"], 1.0)

    def test_short_foreign_quote_does_not_fail(self) -> None:
        response = " ".join(["paraula"] * 80) + "\n\nHola amigo."

        with mock.patch(
            "lm_eval_tasks.catalan_drift.utils._predict_fasttext",
            side_effect=fake_fasttext,
        ):
            result = process_results({"target_lang": "ca", "category": "test"}, [response])

        self.assertEqual(result["language_fail"], 0.0)
        self.assertEqual(result["drift_pass"], 1.0)

    def test_short_labeled_catalan_template_does_not_fail(self) -> None:
        lines = [
            "ASSUMPTE: Retard del lliurament",
            "SITUACIÓ: Estoc insuficient",
            "RISC: Penalització contractual",
            "ACCIÓ: Comanda urgent",
            "SEGÜENT PAS: Trucada dilluns",
        ]

        for separator in ("\n", "\n\n"):
            with self.subTest(separator=repr(separator)):
                with mock.patch(
                    "lm_eval_tasks.catalan_drift.utils._predict_fasttext",
                    return_value=("ca", 0.99),
                ) as predict:
                    result = process_results(
                        {"target_lang": "ca", "category": "test"},
                        [separator.join(lines)],
                    )

                self.assertEqual(result["language_fail"], 0.0)
                self.assertEqual(result["drift_pass"], 1.0)
                predict.assert_called_once()

    def test_detector_exception_is_not_silent_pass(self) -> None:
        with mock.patch(
            "lm_eval_tasks.catalan_drift.utils._predict_fasttext",
            side_effect=RuntimeError("detector exploded"),
        ):
            result = process_results(
                {"target_lang": "ca", "category": "test"},
                ["Aquesta resposta sembla catalana i hauria de requerir detector."],
            )

        self.assertEqual(result["language_fail"], 1.0)
        self.assertEqual(result["drift_pass"], 0.0)


class LcbLineLanguageTest(unittest.TestCase):
    def test_any_incorrect_eligible_line_fails_response(self) -> None:
        response = (
            "Aquesta resposta catalana conté prou paraules per classificar-la.\n"
            "Esta línea está escrita completamente en español ahora."
        )

        with mock.patch(
            "lm_eval_tasks.catalan_drift.utils._predict_fasttext",
            side_effect=[("ca", 0.99), ("es", 0.98)],
        ):
            result = lcb_line_language_result({"target_lang": "ca"}, response)

        self.assertTrue(result["eligible"])
        self.assertFalse(result["passed"])
        self.assertEqual(result["eligible_lines"], 2)
        self.assertEqual(result["error_lines"], 1)
        self.assertEqual(result["line_accuracy"], 0.5)

    def test_short_lines_are_excluded_like_official_lcb(self) -> None:
        response = "Hola amigo.\nAquesta resposta catalana té més de cinc paraules."

        with mock.patch(
            "lm_eval_tasks.catalan_drift.utils._predict_fasttext",
            return_value=("ca", 0.99),
        ) as predict:
            result = lcb_line_language_result({"target_lang": "ca"}, response)

        self.assertTrue(result["passed"])
        self.assertEqual(result["eligible_lines"], 1)
        predict.assert_called_once_with("Aquesta resposta catalana té més de cinc paraules")

    def test_low_confidence_prediction_becomes_unknown(self) -> None:
        with mock.patch(
            "lm_eval_tasks.catalan_drift.utils._predict_fasttext",
            return_value=("ca", 0.3),
        ):
            result = lcb_line_language_result(
                {"target_lang": "ca"},
                "Aquesta línia catalana conté exactament prou paraules.",
            )

        self.assertFalse(result["passed"])
        self.assertEqual(result["lines"][0]["predicted_lang"], "unknown")

    def test_response_without_eligible_lines_is_skipped(self) -> None:
        with mock.patch(
            "lm_eval_tasks.catalan_drift.utils._predict_fasttext",
        ) as predict:
            result = lcb_line_language_result({"target_lang": "ca"}, "Massa curt.")

        self.assertFalse(result["eligible"])
        self.assertIsNone(result["passed"])
        self.assertIsNone(result["line_accuracy"])
        predict.assert_not_called()

    def test_q_suffix_is_removed_before_line_detection(self) -> None:
        response = (
            "Aquesta és una resposta catalana completa i adequada.\n"
            "Q: This continuation must not be evaluated by LCB."
        )

        with mock.patch(
            "lm_eval_tasks.catalan_drift.utils._predict_fasttext",
            return_value=("ca", 0.99),
        ) as predict:
            result = lcb_line_language_result({"target_lang": "ca"}, response)

        self.assertTrue(result["passed"])
        self.assertEqual(result["eligible_lines"], 1)
        predict.assert_called_once()


if __name__ == "__main__":
    unittest.main()
