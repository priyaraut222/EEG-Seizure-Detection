"""Reading the 2.65 GB CSV without loading it into RAM.

Strategy: read the CSV ONCE in chunks. During that single pass we validate it, collect summary
statistics and write a float32 .npy copy that later phases open as a memory map (fast random access).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import (CACHE_META, CSV_PATH, LABEL_CACHE, SIGNAL_CACHE, TARGET_COLUMN, ensure_dirs, load_json)


def check_csv_exists(path: Path = CSV_PATH) -> Path:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download EEG_Scaled_data.csv from Kaggle "
            "(subirbiswas19/eeg-dataset) and place it in the project root."
        )
    return Path(path)


def read_head(n_rows: int = 5, path: Path = CSV_PATH) -> pd.DataFrame:
    """Read only the first few rows (cheap first look)."""
    return pd.read_csv(check_csv_exists(path), nrows=n_rows)


def validate_header(columns) -> dict:
    """Check that the header is Channel_1 ... Channel_N followed by `target`, in that order."""
    columns = list(columns)
    problems = []
    if columns[-1] != TARGET_COLUMN:
        problems.append(f"last column is '{columns[-1]}', expected '{TARGET_COLUMN}'")
    signal_cols = columns[:-1]
    expected = [f"Channel_{i}" for i in range(1, len(signal_cols) + 1)]
    if signal_cols != expected:
        problems.append("signal columns are not exactly Channel_1..Channel_N in order")
    return {"n_columns": len(columns), "n_signal_columns": len(signal_cols), "problems": problems}


def count_data_rows(path: Path = CSV_PATH, block_bytes: int = 64 * 1024 * 1024) -> int:
    """Count data rows by counting newlines (fast, no parsing). Assumes no newlines inside fields."""
    n_newlines, last_byte = 0, b"\n"
    with open(check_csv_exists(path), "rb") as f:
        while block := f.read(block_bytes):
            n_newlines += block.count(b"\n")
            last_byte = block[-1:]
    n_lines = n_newlines + (0 if last_byte == b"\n" else 1)
    return n_lines - 1  # minus header


def cache_is_ready() -> bool:
    return CACHE_META.exists() and SIGNAL_CACHE.exists() and LABEL_CACHE.exists()


def build_cache_and_summary(path: Path = CSV_PATH, chunk_rows: int = 250, force: bool = False) -> dict:
    """One chunked pass over the CSV. Returns (and caches) a dataset summary.

    Writes: signals_float32.npy, labels.npy, cache_meta.json (the summary) under data/processed/.
    """
    ensure_dirs()
    if cache_is_ready() and not force:
        return load_json(CACHE_META)

    path = check_csv_exists(path)
    n_rows = count_data_rows(path)
    header = list(pd.read_csv(path, nrows=0).columns)
    header_report = validate_header(header)
    if header_report["problems"]:
        raise ValueError("Unexpected CSV header: " + "; ".join(header_report["problems"]))
    n_signal = header_report["n_signal_columns"]

    signals = np.lib.format.open_memmap(SIGNAL_CACHE, mode="w+", dtype=np.float32, shape=(n_rows, n_signal))
    labels = np.empty(n_rows, dtype=np.int64)

    col_sum = np.zeros(n_signal)        # float64 running sums -> column mean / std
    col_sumsq = np.zeros(n_signal)
    n_nan = n_inf = 0
    row_hashes = []
    row_pos = 0

    for chunk in pd.read_csv(path, chunksize=chunk_rows):
        non_numeric = [c for c in chunk.columns if not pd.api.types.is_numeric_dtype(chunk[c])]
        if non_numeric:
            raise ValueError(f"Non-numeric columns found (first 5): {non_numeric[:5]}")
        values = chunk[header[:-1]].to_numpy(dtype=np.float64)
        target = chunk[TARGET_COLUMN]
        if target.isna().any():
            raise ValueError(f"Missing target values in rows {row_pos}-{row_pos + len(chunk)}")
        if not np.allclose(target, np.round(target)):
            raise ValueError("Target contains non-integer values")

        n_nan += int(np.isnan(values).sum())
        n_inf += int(np.isinf(values).sum())
        clean = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)  # only affects the sums below
        col_sum += clean.sum(axis=0)
        col_sumsq += (clean ** 2).sum(axis=0)
        row_hashes.append(pd.util.hash_pandas_object(chunk, index=False).to_numpy())

        end = row_pos + len(chunk)
        signals[row_pos:end] = values.astype(np.float32)
        labels[row_pos:end] = np.round(target).astype(np.int64)
        row_pos = end

    if row_pos != n_rows:
        raise ValueError(f"Row count mismatch: counted {n_rows} newlines but parsed {row_pos} rows")
    signals.flush()
    np.save(LABEL_CACHE, labels)

    col_mean = col_sum / n_rows
    col_std = np.sqrt(np.maximum(col_sumsq / n_rows - col_mean ** 2, 0.0))
    all_hashes = np.concatenate(row_hashes)
    values_, counts_ = np.unique(labels, return_counts=True)

    summary = {
        "csv_file": Path(path).name,
        "csv_size_gb": round(Path(path).stat().st_size / 1e9, 3),
        "n_rows": int(n_rows),
        "n_columns": int(header_report["n_columns"]),
        "n_signal_columns": int(n_signal),
        "header_problems": header_report["problems"],
        "all_signal_columns_numeric": True,   # a non-numeric column raises above
        "n_nan_values": n_nan,
        "n_inf_values": n_inf,
        "n_duplicate_rows": int(len(all_hashes) - len(np.unique(all_hashes))),
        "label_counts": {int(v): int(c) for v, c in zip(values_, counts_)},
        "column_mean_min": float(col_mean.min()), "column_mean_max": float(col_mean.max()),
        "column_std_min": float(col_std.min()), "column_std_max": float(col_std.max()),
        "n_constant_columns": int((col_std < 1e-12).sum()),
        "signal_min": float(signals.min()), "signal_max": float(signals.max()),
    }
    with open(CACHE_META, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    np.save(SIGNAL_CACHE.with_name("column_mean.npy"), col_mean)
    np.save(SIGNAL_CACHE.with_name("column_std.npy"), col_std)
    return summary


def open_signals() -> np.ndarray:
    """Memory-mapped (n_rows, n_signal_columns) float32 array. Nothing is loaded until sliced."""
    if not cache_is_ready():
        raise FileNotFoundError("Signal cache missing. Run notebooks/01_data_inspection.ipynb first (it builds it).")
    return np.load(SIGNAL_CACHE, mmap_mode="r")


def load_labels() -> np.ndarray:
    if not LABEL_CACHE.exists():
        raise FileNotFoundError("Label cache missing. Run notebooks/01_data_inspection.ipynb first.")
    return np.load(LABEL_CACHE)


def infer_label_mapping(label_counts: dict, reported_seizure: int, reported_non_seizure: int) -> dict:
    """Decide which label value means 'seizure'.

    The Kaggle page does not state the meaning of `target`. The only documentation we have is the
    creator's class counts, so we match observed counts to reported counts and label the result
    'inferred' rather than 'documented'.
    """
    counts = {int(k): int(v) for k, v in label_counts.items()}
    if len(counts) != 2:
        return {"seizure_label": None, "status": f"expected 2 classes, found {sorted(counts)}"}
    seizure = [k for k, v in counts.items() if v == reported_seizure]
    non_seizure = [k for k, v in counts.items() if v == reported_non_seizure]
    if len(seizure) == 1 and len(non_seizure) == 1 and seizure[0] != non_seizure[0]:
        return {"seizure_label": seizure[0], "non_seizure_label": non_seizure[0],
                "status": "inferred: observed class counts match the counts reported by the dataset creator"}
    minority = min(counts, key=counts.get)
    return {"seizure_label": minority, "non_seizure_label": max(counts, key=counts.get),
            "status": "UNVERIFIED: counts differ from the reported ones; minority class assumed to be seizure"}
