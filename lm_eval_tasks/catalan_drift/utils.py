import os
import re
import shutil
import string
import subprocess
from collections.abc import Iterable
from functools import partial
from pathlib import Path

from lm_eval.api.task import ConfigurableTask
from mantinc import dataset_path

LANGUAGE_FAIL_NON_CA_RATIO = 0.15
LANGUAGE_MIN_CONFIDENCE = 0.65
LANGUAGE_MIN_ALPHA_TOKENS = 5
LANGUAGE_WINDOW_TOKENS = 40
LCB_LINE_MIN_TOKENS = 5
LCB_LINE_MIN_CONFIDENCE = 0.3
DEFAULT_LANGUAGE_ID_MODEL = "models/lid.176.ftz"
DEFAULT_FASTTEXT_BINARY = "models/fasttext"

_ALPHA_TOKEN_RE = re.compile(r"[^\W\d_]+(?:[.'’·-][^\W\d_]+)*", re.UNICODE)
_CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)


def first_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("response", "text", "content", "output"):
            text = first_text(value.get(key))
            if text:
                return text
    if isinstance(value, (list, tuple)):
        for item in value:
            text = first_text(item)
            if text:
                return text
    return ""


class CatalanDriftTask(ConfigurableTask):
    def __init__(self, config=None):
        if shutil.which(_fasttext_binary()) is None:
            raise RuntimeError("fastText executable not found; install fasttext before running evaluations")
        if config:
            config = {key: value for key, value in config.items() if key != "class"}
            config["dataset_path"] = "json"
            config["dataset_name"] = None
            config["dataset_kwargs"] = {"data_files": {"test": str(dataset_path())}}
        super().__init__(config=config)

    def fewshot_context(
        self,
        doc,
        num_fewshot,
        system_instruction=None,
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        chat_template=None,
        gen_prefix=None,
    ):
        if not doc.get("messages"):
            return super().fewshot_context(
                doc,
                num_fewshot,
                system_instruction=system_instruction,
                apply_chat_template=apply_chat_template,
                fewshot_as_multiturn=fewshot_as_multiturn,
                chat_template=chat_template,
                gen_prefix=gen_prefix,
            )

        messages = list(doc["messages"])
        if system_instruction:
            if messages and messages[0].get("role") == "system":
                messages[0] = {
                    "role": "system",
                    "content": system_instruction + "\n\n" + messages[0]["content"],
                }
            else:
                messages.insert(0, {"role": "system", "content": system_instruction})

        if apply_chat_template and chat_template:
            return partial(chat_template, add_generation_prompt=not gen_prefix)(messages)

        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)


def _alpha_tokens(text: str) -> list[str]:
    return _ALPHA_TOKEN_RE.findall(_URL_RE.sub(" ", text))


def _language_segments(text: str) -> Iterable[tuple[str, int]]:
    text = _CODE_FENCE_RE.sub("\n\n", text)
    fallback_tokens = []
    yielded = False
    for block in re.split(r"\n\s*\n+", text):
        block = block.strip()
        if not block:
            continue
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        units = lines if len(lines) > 1 else _SENTENCE_BOUNDARY_RE.split(block)
        for unit in units:
            tokens = _alpha_tokens(unit)
            if len(tokens) < LANGUAGE_MIN_ALPHA_TOKENS:
                fallback_tokens.extend(tokens)
                continue
            if len(tokens) <= LANGUAGE_WINDOW_TOKENS:
                yielded = True
                yield _URL_RE.sub(" ", unit), len(tokens)
                continue
            for start in range(0, len(tokens), LANGUAGE_WINDOW_TOKENS):
                window = tokens[start : start + LANGUAGE_WINDOW_TOKENS]
                if len(window) >= LANGUAGE_MIN_ALPHA_TOKENS:
                    yielded = True
                    yield " ".join(window), len(window)
    if not yielded and len(fallback_tokens) >= LANGUAGE_MIN_ALPHA_TOKENS:
        yield " ".join(fallback_tokens), len(fallback_tokens)


def _fasttext_binary() -> str:
    binary = os.environ.get("LANGUAGE_ID_FASTTEXT_BIN")
    if binary:
        return binary
    if Path(DEFAULT_FASTTEXT_BINARY).exists():
        return DEFAULT_FASTTEXT_BINARY
    return shutil.which("fasttext") or "fasttext"


def _predict_fasttext(text: str) -> tuple[str, float]:
    model = os.environ.get("LANGUAGE_ID_MODEL", DEFAULT_LANGUAGE_ID_MODEL)
    result = subprocess.run(
        [_fasttext_binary(), "predict-prob", model, "-", "1"],
        input=text.replace("\n", " ") + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    parts = result.stdout.strip().split()
    if len(parts) < 2:
        raise RuntimeError("fastText did not return a prediction")
    lang = parts[0].removeprefix("__label__").casefold().replace("-", "_").split("_", 1)[0]
    return lang, float(parts[1])


def _lcb_normalize(text: str) -> str:
    """Reproduce the normalization in the official LCB metric script."""
    text = text.split("\nQ:", 1)[0].strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.replace("—", " ").replace("،", "")


def lcb_line_language_result(doc, response: str) -> dict[str, object]:
    """Apply Marchisio et al.'s released line-level detector logic.

    LCB normalizes punctuation, splits on newlines, skips lines with fewer
    than five whitespace-delimited tokens, and fails if any eligible line's
    top fastText label differs from the target. Predictions at confidence
    <= 0.3 become ``unknown`` and therefore fail.

    Responses without eligible lines are excluded from LCB's LPR denominator,
    represented here by ``passed=None``.
    """
    target_lang = str(doc.get("target_lang") or "").casefold()
    if not target_lang:
        raise ValueError("LCB line detection requires doc['target_lang']")

    lines = []
    for line_number, line in enumerate(_lcb_normalize(response).split("\n"), 1):
        tokens = line.split()
        if len(tokens) < LCB_LINE_MIN_TOKENS:
            continue
        lang, confidence = _predict_fasttext(line)
        predicted_lang = lang if confidence > LCB_LINE_MIN_CONFIDENCE else "unknown"
        lines.append(
            {
                "line_number": line_number,
                "text": line,
                "tokens": len(tokens),
                "predicted_lang": predicted_lang,
                "raw_predicted_lang": lang,
                "confidence": confidence,
                "error": predicted_lang != target_lang,
            }
        )

    if not lines:
        return {
            "eligible": False,
            "passed": None,
            "line_accuracy": None,
            "eligible_lines": 0,
            "error_lines": 0,
            "lines": [],
        }

    error_lines = sum(bool(line["error"]) for line in lines)
    return {
        "eligible": True,
        "passed": error_lines == 0,
        "line_accuracy": 1 - error_lines / len(lines),
        "eligible_lines": len(lines),
        "error_lines": error_lines,
        "lines": lines,
    }


def _language_token_counts(doc, response):
    if doc.get("target_lang") != "ca" or not response.strip():
        return 0, 0, []

    total_tokens = 0
    non_catalan_tokens = 0
    try:
        for segment, tokens in _language_segments(response):
            total_tokens += tokens
            lang, confidence = _predict_fasttext(segment)
            if confidence >= LANGUAGE_MIN_CONFIDENCE and lang != "ca":
                non_catalan_tokens += tokens
    except Exception as exc:
        return total_tokens, non_catalan_tokens, [f"detector: {exc}"]

    if total_tokens == 0:
        return total_tokens, non_catalan_tokens, ["detector: no detectable language tokens"]

    return total_tokens, non_catalan_tokens, []


def _language_errors_from_counts(total_tokens, non_catalan_tokens):
    if total_tokens == 0:
        return []
    ratio = non_catalan_tokens / total_tokens
    if ratio >= LANGUAGE_FAIL_NON_CA_RATIO:
        return [f"non_catalan_token_ratio={ratio:.3f}"]
    return []


def _language_errors(doc, response):
    total_tokens, non_catalan_tokens, errors = _language_token_counts(doc, response)
    return errors or _language_errors_from_counts(total_tokens, non_catalan_tokens)


def catalan_token_ratio(items):
    catalan_tokens = sum(item[0] for item in items)
    total_tokens = sum(item[1] for item in items)
    return catalan_tokens / total_tokens if total_tokens else 0.0


def process_results(doc, results):
    response = first_text(results).strip()
    total_tokens, non_catalan_tokens, detector_errors = _language_token_counts(doc, response)
    language_errors = detector_errors or _language_errors_from_counts(total_tokens, non_catalan_tokens)
    api_or_empty_fail = not response
    passed = not (api_or_empty_fail or language_errors)
    category = str(doc.get("category", "unknown"))
    catalan_tokens = total_tokens - non_catalan_tokens
    return {
        "drift_pass": float(passed),
        "language_fail": float(bool(language_errors)),
        "api_or_empty_fail": float(api_or_empty_fail),
        "catalan_token_ratio": (catalan_tokens, total_tokens),
        f"{category}_pass": float(passed),
    }
