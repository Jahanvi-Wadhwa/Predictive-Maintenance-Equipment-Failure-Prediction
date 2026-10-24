"""Evaluation utilities: metric computation and chart generation.

All functions work on real predictions (no synthetic numbers) so they can be
reused by the training script, the tests and the dashboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

def trapz(y: Any, x: Any) -> float:
    """Trapezoidal integration compatible with numpy 1.x and 2.x."""
    fn = getattr(np, "trapezoid", None)
    if fn is None:  # numpy < 2.0
        fn = np.trapz  # type: ignore[attr-defined]
    return float(fn(y, x))


# Backwards-compatible private alias.
_trapz = trapz


PALETTE = {
    "navy": "#0f2a4a",
    "slate": "#475569",
    "teal": "#0d9488",
    "teal_light": "#5eead4",
    "white": "#ffffff",
    "light": "#e2e8f0",
    "alert": "#d97706",
}


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, Any]:
    """Compute the full metric suite used throughout the project.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels (threshold 0.5).
        y_proba: Predicted probabilities of the positive (failure) class.

    Returns:
        Dict with accuracy, precision, recall, F1, ROC-AUC, average precision
        and the confusion matrix. ROC-AUC/AP fall back to ``None`` when only
        one class is present in ``y_true`` (e.g. tiny test slices).
    """
    single_class = len(np.unique(y_true)) < 2
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None if single_class else float(roc_auc_score(y_true, y_proba)),
        "average_precision": (
            None if single_class else float(average_precision_score(y_true, y_proba))
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "n_samples": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }


def curve_data(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, list]:
    """Return ROC and precision-recall curve points for plotting/shipping."""
    if len(np.unique(y_true)) < 2:
        return {"fpr": [], "tpr": [], "thresholds_roc": [],
                "precision": [], "recall": [], "thresholds_pr": []}
    fpr, tpr, roc_thr = roc_curve(y_true, y_proba)
    precision, recall, pr_thr = precision_recall_curve(y_true, y_proba)
    return {
        "fpr": np.round(fpr, 6).tolist(),
        "tpr": np.round(tpr, 6).tolist(),
        "thresholds_roc": np.round(roc_thr, 6).tolist(),
        "precision": np.round(precision, 6).tolist(),
        "recall": np.round(recall, 6).tolist(),
        "thresholds_pr": np.round(pr_thr, 6).tolist(),
    }


def _style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.7)


def plot_confusion_matrix(cm: list[list[int]], title: str, path: Path) -> Path:
    matrix = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(matrix, cmap="Blues")
    labels = ["Predicted\nno failure", "Predicted\nfailure"]
    ax.set_xticks([0, 1], labels, fontsize=9)
    ax.set_yticks([0, 1], ["Actual\nno failure", "Actual\nfailure"], fontsize=9)
    thresh = matrix.max() / 2 if matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if matrix[i, j] > thresh else PALETTE["navy"])
    ax.set_title(title, fontweight="bold", color=PALETTE["navy"], fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_curves(curves: dict[str, dict[str, list]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = [PALETTE["navy"], PALETTE["teal"], PALETTE["alert"], PALETTE["slate"]]
    for i, (name, c) in enumerate(curves.items()):
        auc = roc_auc_of(c)
        ax.plot(c["fpr"], c["tpr"], color=colors[i % len(colors)],
                linewidth=1.8, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=PALETTE["slate"], linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves on the held-out test set",
                 fontweight="bold", color=PALETTE["navy"])
    ax.legend(frameon=False, fontsize=9)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def roc_auc_of(curve: dict[str, list]) -> float:
    """Trapezoidal ROC-AUC from stored curve points (used for plot legends)."""
    fpr, tpr = np.asarray(curve["fpr"]), np.asarray(curve["tpr"])
    if len(fpr) < 2:
        return float("nan")
    return _trapz(tpr, fpr)


def plot_pr_curves(
    curves: dict[str, dict[str, list]], baseline: float, path: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = [PALETTE["navy"], PALETTE["teal"], PALETTE["alert"], PALETTE["slate"]]
    for i, (name, c) in enumerate(curves.items()):
        ap = _trapz(c["precision"], c["recall"]) if len(c["recall"]) > 1 else float("nan")
        ax.plot(c["recall"], c["precision"], color=colors[i % len(colors)],
                linewidth=1.8, label=f"{name} (AP = {ap:.3f})")
    ax.axhline(baseline, linestyle="--", color=PALETTE["slate"], linewidth=1,
               label=f"Baseline (failure rate = {baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves on the held-out test set",
                 fontweight="bold", color=PALETTE["navy"])
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_model_comparison(cv_results: dict[str, dict], path: Path) -> Path:
    """Bar chart of mean CV ROC-AUC and average precision per model."""
    names = list(cv_results)
    roc = [cv_results[n]["roc_auc_mean"] for n in names]
    ap = [cv_results[n]["average_precision_mean"] for n in names]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    b1 = ax.bar(x - width / 2, roc, width, label="ROC-AUC (CV)", color=PALETTE["navy"])
    b2 = ax.bar(x + width / 2, ap, width, label="Avg. precision (CV)", color=PALETTE["teal"])
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, names, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score (5-fold CV on training set)")
    ax.set_title("Model comparison - cross-validation",
                 fontweight="bold", color=PALETTE["navy"])
    ax.legend(frameon=False, fontsize=9)
    _style_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
