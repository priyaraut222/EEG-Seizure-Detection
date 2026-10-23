"""Model building, splitting and training (scikit-learn). Positive class (1) = seizure."""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .utils import RANDOM_SEED


def load_feature_table(path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run notebooks/03_feature_extraction.ipynb first.")
    return pd.read_csv(path)


def prepare_xy(table: pd.DataFrame, seizure_label: int, drop_columns=()):
    """Return features X, binary target y (1 = seizure) and the original row index."""
    X = table.drop(columns=["row_index", "target", *drop_columns])
    if not np.isfinite(X.to_numpy(dtype=np.float64)).all():
        raise ValueError("Feature table has NaN/inf values. Fix them in Phase 3 before training.")
    y = (table["target"] == seizure_label).astype(int).to_numpy()
    return X, y, table["row_index"].to_numpy()


def stratified_indices(y, test_size: float = 0.2, seed: int = RANDOM_SEED):
    """Random stratified split of SEGMENTS, returned as positional index arrays (train_idx, test_idx).

    No patient/recording ids exist in this dataset, so this is the best split available.
    """
    return train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed)


def ordered_split(X, y, test_fraction: float = 0.2):
    """First (1-f) of the rows -> train, last f -> test. Exploratory leakage check only (see Phase 4)."""
    cut = int(len(X) * (1 - test_fraction))
    return X.iloc[:cut], X.iloc[cut:], y[:cut], y[cut:]


def build_models(seed: int = RANDOM_SEED) -> dict:
    """Scaler lives INSIDE each pipeline, so it is fitted on training data only (no leakage).

    class_weight='balanced' makes mistakes on the rare seizure class cost more (imbalance handling).
    """
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=seed)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(
            kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=seed)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
    }


def cross_validate_models(models: dict, X_train, y_train, folds: int = 5, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Stratified k-fold on the TRAINING set only; mean and std of each metric per model."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scoring = {"recall": "recall", "precision": "precision", "f1": "f1", "pr_auc": "average_precision", "roc_auc": "roc_auc"}
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        row = {"model": name}
        for key in scoring:
            row[f"{key}_mean"] = scores[f"test_{key}"].mean()
            row[f"{key}_std"] = scores[f"test_{key}"].std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def get_scores(model, X) -> np.ndarray:
    """Continuous seizure score for ROC/PR curves (probability if available, else decision function)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def save_model(model, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
