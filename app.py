"""Optional demo: inspect one segment and see a trained model's output.

Run:  uv run --group app streamlit run app/streamlit_app.py
Needs Phases 1-4 to have been run (signal cache, features.csv and models/*.joblib).
Educational demo only - not a medical device.
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from eeg_signal_classification import data_loader as dl
from eeg_signal_classification import feature_extraction as fe
from eeg_signal_classification import signal_analysis as sa
from eeg_signal_classification.train import get_scores
from eeg_signal_classification.utils import FEATURE_FILE, MODELS_DIR, load_structure

st.set_page_config(page_title="EEG seizure detection demo", layout="wide")
st.title("EEG seizure detection - demo")
st.warning("Educational demo. Not clinically validated; the random segment-level split cannot show patient-independent performance.")

model_files = sorted(MODELS_DIR.glob("*.joblib"))
if not model_files or not FEATURE_FILE.exists() or not dl.cache_is_ready():
    st.error("Run notebooks 01-04 first (signal cache, features.csv and trained models are required).")
    st.stop()


@st.cache_resource
def load_everything():
    return load_structure(), dl.open_signals(), dl.load_labels(), list(pd.read_csv(FEATURE_FILE, nrows=1).columns)


structure, signals, labels, feature_table_columns = load_everything()
feature_columns = [c for c in feature_table_columns if c not in ("row_index", "target")]
fs_hz = structure["fs_hz"]

model_name = st.sidebar.selectbox("Model", [p.stem for p in model_files])
row = st.sidebar.number_input("Row index", min_value=0, max_value=len(labels) - 1, value=0, step=1)
if st.sidebar.button("Random seizure row"):
    row = int(pd.Series((labels == structure["seizure_label"]).nonzero()[0]).sample(1).iloc[0])
    st.sidebar.write(f"Using row {row}")

model = joblib.load(MODELS_DIR / f"{model_name}.joblib")
x = signals[row].astype("float64")
features = fe.extract_chunk(x[None, :], structure, fs_hz)[feature_columns]   # same columns/order as training
score = float(get_scores(model, features)[0])
predicted_seizure = bool(model.predict(features)[0])
true_seizure = labels[row] == structure["seizure_label"]

left, right = st.columns(2)
left.metric("Model prediction", "seizure" if predicted_seizure else "non-seizure", help=f"score = {score:.3f}")
right.metric("Label in the dataset", "seizure" if true_seizure else "non-seizure",
             help=f"label mapping status: {structure['label_status']}")

st.subheader("Signal and spectrum")
if structure["layout"] in ("blocks", "single_sequence"):
    block = st.sidebar.number_input("Block", 1, structure["n_blocks"], 1) - 1
    segment = x.reshape(structure["n_blocks"], structure["block_length"])[block]
    freqs, psd = sa.welch_psd(segment, sa.effective_fs(fs_hz), 512)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].plot(segment, lw=0.7); axes[0].set(xlabel="sample index", ylabel="z-scored amplitude", title=f"Block {block + 1}")
    axes[1].semilogy(freqs, psd); axes[1].set(xlabel=f"frequency ({sa.freq_unit(fs_hz)})", ylabel="PSD", title="Welch PSD")
    st.pyplot(fig)
else:
    st.info("Row structure was not verified in Phase 1, so no waveform or spectrum is shown.")
st.caption(f"layout: {structure['layout']} | sampling rate: {structure['fs_status']}")
