"""Checks on what the dataset creator already did to the data, and the row-structure investigation.

Main question of Phase 1: is each row really a time series, and if so where do channels / segments start?
We cannot read this from the CSV header, so we infer it from the data itself using one simple idea:

    For a true waveform, neighbouring columns are strongly correlated (the signal is smooth).
    For columns that are unrelated (shuffled or independent features), they are not.

    Let r_j = E[(x_{j+1} - x_j)^2] / (Var(x_j) + Var(x_{j+1})).
      r_j ~ 1  -> columns j and j+1 are independent (no waveform continuity)
      r_j ~ 1 - rho  -> neighbouring samples with correlation rho (smaller r = smoother signal)
    If a row is several channels stored back-to-back, r_j jumps up at every channel boundary,
    because the last sample of one channel and the first sample of the next are unrelated.
"""

import numpy as np


def check_standard_scaling(summary: dict, tol_mean: float = 0.05, tol_std: float = 0.05) -> dict:
    """Do the per-COLUMN means/stds look like sklearn's StandardScaler output (mean 0, std 1)?"""
    mean_ok = max(abs(summary["column_mean_min"]), abs(summary["column_mean_max"])) < tol_mean
    std_ok = (abs(summary["column_std_min"] - 1) < tol_std) and (abs(summary["column_std_max"] - 1) < tol_std)
    return {"column_means_near_0": bool(mean_ok), "column_stds_near_1": bool(std_ok),
            "consistent_with_column_wise_standard_scaling": bool(mean_ok and std_ok)}


def adjacent_column_profile(X: np.ndarray, column_chunk: int = 4096) -> np.ndarray:
    """r_j for every neighbouring column pair, estimated over the rows of X (see module docstring).

    Works on slices of columns so memory stays small even for 36,864 columns.
    """
    n_columns = X.shape[1]
    profile = np.empty(n_columns - 1)
    for start in range(0, n_columns - 1, column_chunk):
        stop = min(start + column_chunk, n_columns - 1)
        part = np.asarray(X[:, start:stop + 1], dtype=np.float64)   # one extra column for the last pair
        part = part - part.mean(axis=0)
        var = part.var(axis=0)
        diff_sq = np.mean((part[:, 1:] - part[:, :-1]) ** 2, axis=0)
        profile[start:stop] = diff_sq / np.maximum(var[1:] + var[:-1], 1e-12)
    return profile


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median)) * 1.4826
    return (values - median) / max(mad, 1e-9)


def find_boundary_candidates(profile: np.ndarray, z_threshold: float = 6.0, merge_gap: int = 3,
                             edge_margin: int = 8) -> np.ndarray:
    """Pair indices j (boundary between column j and j+1, 0-based) where r_j is an upward outlier.

    Real segment starts often show a few elevated pairs in a row (filter / scaling edge effects), so
    neighbouring outliers are merged into one boundary at their strongest point. Outliers within
    `edge_margin` columns of the row's ends cannot be a boundary between two blocks and are ignored.
    """
    z = _robust_z(profile)
    raw = np.where(z > z_threshold)[0]
    raw = raw[(raw >= edge_margin) & (raw < len(profile) - edge_margin)]
    if len(raw) == 0:
        return raw
    merged, group = [], [raw[0]]
    for index in raw[1:]:
        if index - group[-1] <= merge_gap:
            group.append(index)
        else:
            merged.append(group[int(np.argmax(profile[group]))])
            group = [index]
    merged.append(group[int(np.argmax(profile[group]))])
    return np.array(merged)


def score_block_length(candidates: np.ndarray, block_length: int, n_columns: int, tolerance: int = 3) -> dict:
    """How well do the detected jumps match a regular grid of boundaries every `block_length` columns?

    A detected jump counts as matching if it lies within `tolerance` columns of a grid boundary.
    """
    if block_length < 2 or n_columns % block_length != 0 or n_columns // block_length < 2:
        return {"block_length": block_length, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    expected = np.arange(block_length - 1, n_columns - 1, block_length)
    found = np.asarray(candidates, dtype=int)
    if len(found) == 0:
        return {"block_length": block_length, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    near = np.abs(found[:, None] - expected[None, :]) <= tolerance
    precision = float(near.any(axis=1).mean())    # detected jumps that sit on the grid
    recall = float(near.any(axis=0).mean())       # grid boundaries that have a detected jump
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"block_length": block_length, "precision": precision, "recall": recall, "f1": f1}


def best_periodic_lag(profile: np.ndarray, max_lag: int = 64) -> tuple[int, float]:
    """Strongest short-lag autocorrelation of the profile (a hint of interleaved channels)."""
    p = profile - profile.mean()
    denom = float(np.dot(p, p)) or 1.0
    best_lag, best_value = 0, 0.0
    for lag in range(2, min(max_lag, len(p) // 4) + 1):
        value = float(np.dot(p[:-lag], p[lag:])) / denom
        if value > best_value:
            best_lag, best_value = lag, value
    return best_lag, best_value


def investigate_structure(X_sample: np.ndarray, min_lag1_corr: float = 0.3) -> dict:
    """Infer how the signal columns of a row are organised. Returns a plain dict (JSON friendly).

    layout values:
      'blocks'            equal-length contiguous segments back-to-back, boundaries detected and regular
      'single_sequence'   smooth/contiguous across the whole row, no boundaries detected
      'unverified'        anything else: do NOT treat columns as a waveform
    """
    n_columns = X_sample.shape[1]
    profile = adjacent_column_profile(X_sample)
    median_r = float(np.median(profile))
    approx_lag1_corr = 1.0 - median_r
    candidates = find_boundary_candidates(profile)

    scores = []
    if len(candidates) >= 1:
        scores.append(score_block_length(candidates, int(candidates[0]) + 1, n_columns))
    if len(candidates) >= 2:
        scores.append(score_block_length(candidates, int(np.median(np.diff(candidates))), n_columns))
    best = max(scores, key=lambda s: s["f1"]) if scores else None
    lag, lag_strength = best_periodic_lag(profile)

    result = {
        "n_rows_used": int(X_sample.shape[0]), "n_columns": int(n_columns),
        "median_adjacent_ratio": median_r, "approx_lag1_correlation": approx_lag1_corr,
        "n_boundary_candidates": int(len(candidates)),
        "boundary_candidates_head": [int(c) for c in candidates[:10]],
        "best_block_fit": best, "short_lag_periodicity": {"lag": lag, "strength": lag_strength},
        "block_length": None, "n_blocks": None,
    }

    contiguous = approx_lag1_corr > min_lag1_corr
    if best is not None and best["precision"] >= 0.9 and best["recall"] >= 0.9 and contiguous:
        result.update(layout="blocks", block_length=best["block_length"], n_blocks=n_columns // best["block_length"],
                      note="Equal-length contiguous segments detected from regular jumps in adjacent-column difference.")
    elif contiguous and len(candidates) == 0 and lag_strength < 0.5:
        result.update(layout="single_sequence", block_length=n_columns, n_blocks=1,
                      note="Columns look like one continuous waveform; no segment boundaries detected. "
                           "A hidden boundary cannot be fully excluded.")
    else:
        reason = "columns are not smooth/continuous" if not contiguous else \
                 "boundaries are irregular or periodicity suggests interleaving"
        result.update(layout="unverified", note=f"Temporal organisation NOT established: {reason}.")
    return result


def sample_rows_evenly(signals: np.ndarray, n_rows: int = 2000, seed: int = 42) -> np.ndarray:
    """Random subset of rows (sorted so the memmap is read in order). Returns an in-memory array."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(signals.shape[0], size=min(n_rows, signals.shape[0]), replace=False))
    return np.asarray(signals[idx])


def high_frequency_check(psd_mean: np.ndarray, freqs: np.ndarray, nyquist: float) -> dict:
    """Is there already a sharp low-pass in the data? Fraction of power in the top 20% of the band."""
    top = freqs >= 0.8 * nyquist
    return {"fraction_power_top_20pct_of_band": float(np.trapezoid(psd_mean[top], freqs[top]) /
                                                      np.trapezoid(psd_mean, freqs))}
