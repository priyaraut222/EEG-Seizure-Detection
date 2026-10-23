import numpy as np
import pytest

from eeg_signal_classification import data_loader, evaluate, feature_extraction as fe, preprocessing as pp
from eeg_signal_classification import signal_analysis as sa, train
from tests.conftest import BLOCK_LEN, N_BLOCKS, RHYTHM_FREQ, make_synthetic


@pytest.fixture
def cache(synthetic_csv, tmp_path, monkeypatch):
    path, data, labels = synthetic_csv
    monkeypatch.setattr(data_loader, "SIGNAL_CACHE", tmp_path / "signals.npy")
    monkeypatch.setattr(data_loader, "LABEL_CACHE", tmp_path / "labels.npy")
    monkeypatch.setattr(data_loader, "CACHE_META", tmp_path / "meta.json")
    summary = data_loader.build_cache_and_summary(path, chunk_rows=64)
    return summary, data, labels


def test_scan_matches_source(cache):
    summary, data, labels = cache
    assert summary["n_rows"] == len(labels) and summary["n_signal_columns"] == data.shape[1]
    assert summary["n_nan_values"] == 0 and summary["n_duplicate_rows"] == 0
    assert sum(summary["label_counts"].values()) == len(labels)
    assert pp.check_standard_scaling(summary)["consistent_with_column_wise_standard_scaling"]
    np.testing.assert_allclose(data_loader.open_signals()[:5], data[:5], atol=1e-4)
    np.testing.assert_array_equal(data_loader.load_labels(), labels)


def test_bad_header_is_rejected():
    assert data_loader.validate_header(["a", "b", "target"])["problems"]


def test_structure_detects_blocks():
    data, _ = make_synthetic()
    result = pp.investigate_structure(data)
    assert result["layout"] == "blocks"
    assert result["block_length"] == BLOCK_LEN and result["n_blocks"] == N_BLOCKS


def test_shuffled_columns_are_not_trusted():
    data, _ = make_synthetic(shuffle_columns=True)
    assert pp.investigate_structure(data)["layout"] == "unverified"


def test_label_mapping():
    ok = data_loader.infer_label_mapping({0: 9799, 1: 1434}, 1434, 9799)
    assert ok["seizure_label"] == 1 and ok["status"].startswith("inferred")
    flipped = data_loader.infer_label_mapping({0: 1434, 1: 9799}, 1434, 9799)
    assert flipped["seizure_label"] == 0
    assert data_loader.infer_label_mapping({0: 10, 1: 90}, 1434, 9799)["status"].startswith("UNVERIFIED")


def test_bandpass_removes_out_of_band_tone():
    t = np.arange(2048)
    low, high = np.sin(2 * np.pi * 0.02 * t), np.sin(2 * np.pi * 0.4 * t)
    out = sa.bandpass_filter(low + high, 0.005, 0.1, fs=1.0)
    assert np.std(out - low) < 0.15 * np.std(high)   # >16 dB suppression of the 0.4 tone


def test_psd_peak_and_band_power():
    t = np.arange(2048)
    x = np.sin(2 * np.pi * 0.1 * t)
    freqs, psd = sa.welch_psd(x, fs=1.0, nperseg=512)
    assert abs(freqs[np.argmax(psd)] - 0.1) < 0.005
    rel = sa.relative_band_power(sa.band_power(freqs, psd, sa.get_bands(None)))
    assert rel["nb1"] + rel["nb2"] > 0.99   # 0.1 sits on the nb1/nb2 edge, so leakage splits it between them
    assert sa.get_bands(256)["gamma"][1] == 80 and "gamma" in sa.get_bands(256)
    assert "gamma" not in sa.get_bands(40)        # 40 Hz fs -> Nyquist 20 Hz, gamma starts at 30


def test_features_and_label_alignment(cache):
    _, data, labels = cache
    structure = pp.investigate_structure(data)
    table = fe.extract_feature_table(data_loader.open_signals(), labels, structure, chunk_rows=100)
    report = fe.quality_report(table)
    assert report["n_nan"] == 0 and report["n_inf"] == 0 and len(table) == len(labels)
    np.testing.assert_array_equal(table["target"].to_numpy(), labels)
    seizure = table[table["target"] == 1]["blk01_dominant_freq"]
    assert abs(seizure.median() - RHYTHM_FREQ) < 0.01          # injected rhythm is recovered
    assert table[table["target"] == 1]["blk01_line_length"].mean() != table[table["target"] == 0]["blk01_line_length"].mean()


def test_statistical_only_mode():
    data, _ = make_synthetic(shuffle_columns=True)
    table = fe.extract_chunk(data, {"layout": "unverified"})
    assert list(table.columns) == [f"row_{n}" for n in fe.STAT_DESCRIPTIONS]


def test_metrics_known_values():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 0])
    m = evaluate.compute_metrics(y_true, y_pred, scores=np.array([.1, .2, .3, .6, .9, .4]))
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 3)
    assert m["specificity"] == pytest.approx(0.75) and m["recall_sensitivity"] == pytest.approx(0.5)
    assert 0 <= m["roc_auc"] <= 1 and 0 <= m["pr_auc"] <= 1


def test_training_pipeline_scaler_inside_and_learns(cache):
    _, data, labels = cache
    structure = pp.investigate_structure(data)
    table = fe.extract_feature_table(data_loader.open_signals(), labels, structure)
    X, y, _ = train.prepare_xy(table, seizure_label=1)
    tr, te = train.stratified_indices(y)
    X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y[tr], y[te]
    assert abs(y_tr.mean() - y_te.mean()) < 0.03   # stratified
    for name, model in train.build_models().items():
        model.fit(X_tr, y_tr)
        metrics = evaluate.compute_metrics(y_te, model.predict(X_te), train.get_scores(model, X_te))
        assert metrics["roc_auc"] > 0.9, name        # easy synthetic task
    assert type(train.build_models()["SVM (RBF)"].steps[0][1]).__name__ == "StandardScaler"
    cv = train.cross_validate_models({"LR": train.build_models()["Logistic Regression"]}, X_tr, y_tr, folds=3)
    assert cv.loc["LR", "roc_auc_mean"] > 0.9
