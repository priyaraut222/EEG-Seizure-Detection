"""Synthetic stand-in for the Kaggle CSV, used ONLY to test the code (never for reported results).

Layout: 4 segments x 256 samples per row, smooth AR(1) noise; seizure rows get a strong 0.08 cycles/sample
rhythm. Columns are standard-scaled like the real file.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.signal import lfilter

N_ROWS, N_BLOCKS, BLOCK_LEN, RHYTHM_FREQ = 300, 4, 256, 0.08


def make_synthetic(seed: int = 0, shuffle_columns: bool = False):
    rng = np.random.default_rng(seed)
    labels = (rng.random(N_ROWS) < 0.2).astype(int)
    burn_in = 200   # discard the filter start-up transient
    noise = lfilter([1.0], [1.0, -0.9], rng.normal(size=(N_ROWS, N_BLOCKS, BLOCK_LEN + burn_in)), axis=-1)[..., burn_in:]
    t = np.arange(BLOCK_LEN)
    phase = rng.uniform(0, 2 * np.pi, size=(N_ROWS, N_BLOCKS, 1))
    rhythm = 2.0 * np.sin(2 * np.pi * RHYTHM_FREQ * t + phase)
    data = (noise + labels[:, None, None] * rhythm).reshape(N_ROWS, -1)
    if shuffle_columns:
        data = data[:, rng.permutation(data.shape[1])]
    data = (data - data.mean(0)) / data.std(0)
    return data, labels


@pytest.fixture
def synthetic_csv(tmp_path):
    data, labels = make_synthetic()
    df = pd.DataFrame(data, columns=[f"Channel_{i}" for i in range(1, data.shape[1] + 1)])
    df["target"] = labels
    path = tmp_path / "synthetic.csv"
    df.to_csv(path, index=False)
    return path, data, labels
