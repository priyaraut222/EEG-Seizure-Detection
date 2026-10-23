"""Digital signal processing building blocks (NumPy / SciPy only).

Frequencies: if the sampling rate is unknown we work in NORMALISED frequency (cycles per sample,
0 to 0.5, i.e. fs = 1). Pass fs_hz to get results in Hz. All functions accept arrays whose LAST axis is time.
"""

import numpy as np
from scipy import signal

CLASSICAL_BANDS_HZ = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0),
                      "beta": (13.0, 30.0), "gamma": (30.0, 80.0)}


def effective_fs(fs_hz) -> float:
    """fs used in calculations: the real one, or 1.0 (normalised frequency)."""
    return float(fs_hz) if fs_hz else 1.0


def freq_unit(fs_hz) -> str:
    return "Hz" if fs_hz else "cycles/sample"


def get_bands(fs_hz) -> dict:
    """Band edges. Hz bands only if fs is known (clipped at Nyquist); otherwise 5 equal normalised bands."""
    if fs_hz:
        nyquist = fs_hz / 2
        return {name: (lo, min(hi, nyquist)) for name, (lo, hi) in CLASSICAL_BANDS_HZ.items() if lo < nyquist}
    edges = np.linspace(0.0, 0.5, 6)
    return {f"nb{i + 1}": (float(edges[i]), float(edges[i + 1])) for i in range(5)}


# ---------- time domain ----------
def time_domain_stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {"mean": x.mean(), "variance": x.var(), "std": x.std(), "rms": np.sqrt(np.mean(x ** 2)),
            "min": x.min(), "max": x.max(), "peak_to_peak": np.ptp(x)}


# ---------- filtering ----------
def butter_bandpass_sos(low: float, high: float, fs: float, order: int = 4):
    """Butterworth band-pass as second-order sections (numerically stable)."""
    nyquist = fs / 2
    if not 0 < low < high < nyquist:
        raise ValueError(f"Need 0 < low < high < Nyquist ({nyquist}); got low={low}, high={high}")
    return signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")


def bandpass_filter(x: np.ndarray, low: float, high: float, fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase filtering (forward-backward), so peaks are not shifted in time."""
    return signal.sosfiltfilt(butter_bandpass_sos(low, high, fs, order), x, axis=-1)


# ---------- frequency domain ----------
def fft_magnitude(x: np.ndarray, fs: float = 1.0):
    """One-sided FFT magnitude |X[k]| / N and its frequency axis."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)  # remove DC so it does not dominate
    spectrum = np.fft.rfft(x, axis=-1)
    return np.fft.rfftfreq(x.shape[-1], d=1.0 / fs), np.abs(spectrum) / x.shape[-1]


def welch_psd(x: np.ndarray, fs: float = 1.0, nperseg: int = 512):
    """Welch PSD: average the periodograms of overlapping Hann-windowed pieces (lower variance)."""
    x = np.asarray(x, dtype=np.float64)
    return signal.welch(x, fs=fs, nperseg=min(nperseg, x.shape[-1]), axis=-1)


def band_power(freqs: np.ndarray, psd: np.ndarray, bands: dict) -> dict:
    """Absolute power per band = area under the PSD curve (trapezoid rule). psd's last axis is frequency."""
    out = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi) if hi < freqs[-1] else (freqs >= lo) & (freqs <= hi)
        out[name] = np.trapezoid(psd[..., mask], freqs[mask], axis=-1)
    return out


def relative_band_power(powers: dict) -> dict:
    """Each band's power divided by the total power over the listed bands."""
    total = sum(powers.values())
    return {name: value / np.maximum(total, 1e-20) for name, value in powers.items()}


def stft_spectrogram(x: np.ndarray, fs: float = 1.0, nperseg: int = 256, overlap: float = 0.5):
    """Short-Time Fourier Transform power spectrogram: freqs, times, |STFT|^2."""
    noverlap = int(nperseg * overlap)
    freqs, times, Sxx = signal.spectrogram(np.asarray(x, dtype=np.float64), fs=fs, nperseg=nperseg,
                                           noverlap=noverlap, mode="psd")
    return freqs, times, Sxx


def mean_psd_by_class(blocks: np.ndarray, labels: np.ndarray, fs: float = 1.0, nperseg: int = 512) -> dict:
    """Median and quartiles of the Welch PSD for each class. blocks: (n, ..., time)."""
    result = {}
    for cls in np.unique(labels):
        freqs, psd = welch_psd(blocks[labels == cls], fs=fs, nperseg=nperseg)
        psd = psd.reshape(-1, psd.shape[-1])   # pool every segment/channel of the class
        result[int(cls)] = {"freqs": freqs, "median": np.median(psd, axis=0),
                            "q25": np.percentile(psd, 25, axis=0), "q75": np.percentile(psd, 75, axis=0)}
    return result
