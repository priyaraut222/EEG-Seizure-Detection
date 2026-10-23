"""Paths, seeds and small helpers shared by every notebook and module."""

import json
import os
import random
from pathlib import Path

import numpy as np


def find_project_root() -> Path:
    """Project folder = nearest parent of this file containing pyproject.toml.

    The environment variable EEG_PROJECT_ROOT can override it (used for tests on synthetic data).
    """
    if os.environ.get("EEG_PROJECT_ROOT"):
        return Path(os.environ["EEG_PROJECT_ROOT"]).resolve()
    for folder in Path(__file__).resolve().parents:
        if (folder / "pyproject.toml").exists():
            return folder
    raise FileNotFoundError("Could not find pyproject.toml above utils.py; is the package installed in editable mode?")


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# The CSV stays in the project root. EEG_CSV_PATH can override it (used by the automated tests).
CSV_PATH = Path(os.environ.get("EEG_CSV_PATH", PROJECT_ROOT / "EEG_Scaled_data.csv"))

TARGET_COLUMN = "target"
RANDOM_SEED = 42

# Everything below was REPORTED by the dataset creator on Kaggle. It is a claim to verify, not a fact.
REPORTED_BY_CREATOR = {
    "n_rows": 11233,
    "n_columns": 36865,
    "n_signal_columns": 36864,
    "n_non_seizure": 9799,
    "n_seizure": 1434,
}

# Cached float32 copy of the signal columns (built once in Phase 1, git-ignored).
SIGNAL_CACHE = PROCESSED_DIR / "signals_float32.npy"
LABEL_CACHE = PROCESSED_DIR / "labels.npy"
CACHE_META = PROCESSED_DIR / "cache_meta.json"
STRUCTURE_FILE = RESULTS_DIR / "structure_decision.json"
FEATURE_FILE = PROCESSED_DIR / "features.csv"


def ensure_dirs() -> None:
    for folder in (PROCESSED_DIR, RESULTS_DIR, MODELS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: Path) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Run the earlier notebook that creates it first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _json_default(value):
    """Let json handle numpy scalars and arrays."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialise {type(value)}")


def load_structure() -> dict:
    """Result of the Phase 1 structure investigation (layout, block length, label mapping, fs)."""
    return load_json(STRUCTURE_FILE)
