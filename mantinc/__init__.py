"""Public integration helpers for the Mantinc benchmark."""

from importlib.resources import files
from pathlib import Path

TASK_NAME = "catalan_drift"


def task_path() -> Path:
    """Return the installed lm-eval task directory."""
    return Path(str(files("lm_eval_tasks")))


def dataset_path() -> Path:
    """Return the dataset snapshot shipped with the installed task."""
    return task_path() / TASK_NAME / "catalan_drift.jsonl"
