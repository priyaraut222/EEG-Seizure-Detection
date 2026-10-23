"""Metrics and plots. Positive class (1) = seizure."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)


def compute_metrics(y_true, y_pred, scores=None) -> dict:
    """Threshold metrics from y_pred; ranking metrics (ROC-AUC, PR-AUC) from continuous scores."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if scores is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, scores)
        metrics["pr_auc"] = average_precision_score(y_true, scores)   # average precision
    return metrics


def plot_confusion_matrices(results: dict, class_names=("non-seizure", "seizure")):
    """results: {model_name: {"y_true":..., "y_pred":...}} -> one counts+row-normalised heatmap each."""
    fig, axes = plt.subplots(1, len(results), figsize=(4.6 * len(results), 4), squeeze=False)
    for ax, (name, res) in zip(axes[0], results.items()):
        cm = confusion_matrix(res["y_true"], res["y_pred"], labels=[0, 1])
        pct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.1%})", ha="center", va="center",
                        color="white" if pct[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1], class_names)
        ax.set_yticks([0, 1], class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(name)
    fig.suptitle("Confusion matrices (test set; percentages are per true class)")
    fig.tight_layout()
    return fig


def plot_model_comparison(metrics_table: pd.DataFrame,
                          columns=("recall_sensitivity", "specificity", "precision", "f1", "pr_auc", "roc_auc")):
    cols = [c for c in columns if c in metrics_table.columns]
    ax = metrics_table[cols].plot(kind="bar", figsize=(10, 4.5), ylim=(0, 1.05), rot=0, width=0.8)
    ax.set_title("Model comparison on the held-out test set")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right", ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    return ax.figure


def plot_roc_pr_curves(results: dict):
    """results: {model_name: {"y_true":..., "scores":...}}"""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["scores"])
        precision, recall, _ = precision_recall_curve(res["y_true"], res["scores"])
        ax_roc.plot(fpr, tpr, label=name)
        ax_pr.plot(recall, precision, label=name)
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate (sensitivity)", title="ROC curves")
    prevalence = float(np.mean(next(iter(results.values()))["y_true"]))
    ax_pr.axhline(prevalence, color="k", ls="--", alpha=0.4, label=f"chance ({prevalence:.2f})")
    ax_pr.set(xlabel="Recall (sensitivity)", ylabel="Precision", title="Precision-recall curves")
    for ax in (ax_roc, ax_pr):
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
