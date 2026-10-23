"""Feature extraction, vectorised over many segments at once.

Which features are allowed depends on what Phase 1 established about the row structure:
  layout 'blocks' / 'single_sequence'  -> statistical + temporal + spectral features (per segment)
  layout 'unverified'                  -> ONLY order-independent statistics of the whole row
                                          (spectral features would be meaningless on arbitrary columns)
"""

import numpy as np
import pandas as pd
from scipy import stats

from .signal_analysis import band_power, effective_fs, get_bands, relative_band_power, welch_psd

STAT_DESCRIPTIONS = {
    "mean": "Average value of the segment (offset).",
    "std": "Standard deviation: typical amplitude fluctuation.",
    "var": "Variance: signal power around the mean.",
    "rms": "Root mean square: overall amplitude including offset.",
    "energy": "Sum of squares: total signal energy of the segment.",
    "ptp": "Peak-to-peak: max minus min, sensitive to large spikes.",
    "skew": "Skewness: asymmetry of the amplitude distribution.",
    "kurtosis": "Excess kurtosis: heavy tails / spikiness of the amplitude distribution.",
}
TEMPORAL_DESCRIPTIONS = {
    "line_length": "Sum of |x[n+1]-x[n]|: grows with amplitude and with fast oscillation; classic seizure feature.",
    "hjorth_mobility": "Hjorth mobility: sqrt(var(dx)/var(x)); proxy for the mean frequency.",
    "hjorth_complexity": "Hjorth complexity: how much the waveform deviates from a pure sine.",
}
SPECTRAL_DESCRIPTIONS = {
    "bp_": "Absolute band power (area under the Welch PSD inside the band).",
    "rbp_": "Relative band power: band power / total power over the bands.",
    "dominant_freq": "Frequency of the largest PSD peak (excluding 0).",
    "spectral_centroid": "Power-weighted mean frequency of the PSD.",
    "spectral_entropy": "Normalised Shannon entropy of the PSD (0 = single tone, 1 = flat/white).",
    "spectral_edge95": "Frequency below which 95% of the power lies.",
}


def statistical_features(x: np.ndarray) -> dict:
    """x: (..., time). Valid for any ordering of the values."""
    return {"mean": x.mean(-1), "std": x.std(-1), "var": x.var(-1), "rms": np.sqrt((x ** 2).mean(-1)),
            "energy": (x ** 2).sum(-1), "ptp": np.ptp(x, axis=-1),
            "skew": stats.skew(x, axis=-1), "kurtosis": stats.kurtosis(x, axis=-1)}


def temporal_features(x: np.ndarray) -> dict:
    """Features that use the ORDER of samples, so need a verified waveform."""
    dx = np.diff(x, axis=-1)
    ddx = np.diff(dx, axis=-1)
    mobility = np.sqrt(dx.var(-1) / np.maximum(x.var(-1), 1e-20))
    mobility_dx = np.sqrt(ddx.var(-1) / np.maximum(dx.var(-1), 1e-20))
    return {"line_length": np.abs(dx).sum(-1), "hjorth_mobility": mobility,
            "hjorth_complexity": mobility_dx / np.maximum(mobility, 1e-20)}


def spectral_features(x: np.ndarray, fs_hz=None, nperseg: int = 512) -> dict:
    """Spectral features from one Welch PSD per segment (computed once and reused)."""
    fs = effective_fs(fs_hz)
    freqs, psd = welch_psd(x, fs=fs, nperseg=nperseg)
    bands = get_bands(fs_hz)
    abs_power = band_power(freqs, psd, bands)
    rel_power = relative_band_power(abs_power)

    positive = freqs > 0
    f, p = freqs[positive], psd[..., positive]
    total = np.maximum(p.sum(-1), 1e-20)
    prob = p / total[..., None]
    cumulative = np.cumsum(prob, axis=-1)

    features = {f"bp_{name}": v for name, v in abs_power.items()}
    features.update({f"rbp_{name}": v for name, v in rel_power.items()})
    features["dominant_freq"] = f[np.argmax(p, axis=-1)]
    features["spectral_centroid"] = (prob * f).sum(-1)
    features["spectral_entropy"] = -(prob * np.log2(np.maximum(prob, 1e-20))).sum(-1) / np.log2(len(f))
    features["spectral_edge95"] = f[np.argmax(cumulative >= 0.95, axis=-1)]
    return features


def feature_mode(structure: dict) -> str:
    return "full" if structure["layout"] in ("blocks", "single_sequence") else "statistical_only"


def extract_chunk(chunk: np.ndarray, structure: dict, fs_hz=None) -> pd.DataFrame:
    """Features for a chunk of rows (n, n_columns) -> DataFrame (n, n_features)."""
    chunk = np.asarray(chunk, dtype=np.float64)
    mode = feature_mode(structure)
    if mode == "full":
        n_blocks, block_len = structure["n_blocks"], structure["block_length"]
        segments = chunk.reshape(len(chunk), n_blocks, block_len)
        prefix = "blk" if structure["layout"] == "blocks" else "seq"
        per_segment = statistical_features(segments)
        per_segment.update(temporal_features(segments))
        per_segment.update(spectral_features(segments, fs_hz))
    else:
        segments = chunk[:, None, :]   # the whole row is one unordered set of values
        prefix = "row"
        per_segment = statistical_features(segments)

    columns = {}
    for name, values in per_segment.items():          # values: (n, n_segments)
        for s in range(values.shape[1]):
            columns[f"{prefix}{s + 1:02d}_{name}" if mode == "full" else f"{prefix}_{name}"] = values[:, s]
    return pd.DataFrame(columns)


def extract_feature_table(signals, labels: np.ndarray, structure: dict, fs_hz=None, chunk_rows: int = 500) -> pd.DataFrame:
    """Process the memory-mapped signals chunk by chunk; keeps row order so labels stay aligned."""
    if len(signals) != len(labels):
        raise ValueError(f"{len(signals)} signal rows but {len(labels)} labels")
    pieces = []
    for start in range(0, len(signals), chunk_rows):
        end = min(start + chunk_rows, len(signals))
        part = extract_chunk(signals[start:end], structure, fs_hz)
        part.insert(0, "target", labels[start:end])
        part.insert(0, "row_index", np.arange(start, end))
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def feature_descriptions(structure: dict, fs_hz=None) -> pd.DataFrame:
    """Human-readable table: which features exist and what they mean."""
    rows = [(k, v, "statistical") for k, v in STAT_DESCRIPTIONS.items()]
    if feature_mode(structure) == "full":
        rows += [(k, v, "temporal") for k, v in TEMPORAL_DESCRIPTIONS.items()]
        unit = "Hz" if fs_hz else "cycles/sample"
        for k, v in SPECTRAL_DESCRIPTIONS.items():
            rows.append((k + "<band>" if k.endswith("_") else k, f"{v} (frequency unit: {unit})", "spectral"))
    return pd.DataFrame(rows, columns=["feature", "meaning", "type"])


def quality_report(table: pd.DataFrame) -> dict:
    """Missing / infinite / constant feature check on the feature columns."""
    feats = table.drop(columns=["row_index", "target"])
    arr = feats.to_numpy(dtype=np.float64)
    return {"n_rows": len(feats), "n_features": feats.shape[1],
            "n_nan": int(np.isnan(arr).sum()), "n_inf": int(np.isinf(arr).sum()),
            "constant_features": [c for c in feats.columns if feats[c].nunique() <= 1]}
