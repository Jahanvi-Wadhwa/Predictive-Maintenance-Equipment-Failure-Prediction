"""Model explainability helpers.

Uses *permutation importance* on the held-out test set - a model-agnostic
method that measures how much each feature's score drops when its values are
randomly shuffled.

Important caveat (also stated in the README and dashboard): feature importance
indicates how much a model *relies* on a feature for its predictions. It does
**not** establish causation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import RANDOM_SEED

NAVY = "#0f2a4a"
TEAL = "#0d9488"
SLATE = "#475569"


def compute_permutation_importance(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    scoring: str = "average_precision",
) -> pd.DataFrame:
    """Compute permutation importance of a fitted pipeline on test data.

    Args:
        pipeline: Fitted imblearn/sklearn pipeline with a ``predict_proba``.
        X_test: Raw (unpreprocessed) test features.
        y_test: Test labels.
        n_repeats: Shuffle repeats (higher = more stable, slower).
        scoring: sklearn scorer name; default average precision because the
            target is imbalanced.

    Returns:
        DataFrame with columns ``feature``, ``importance_mean``,
        ``importance_std`` sorted by mean importance (descending).
    """
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=RANDOM_SEED,
        scoring=scoring,
        n_jobs=1,
    )
    frame = pd.DataFrame(
        {
            "feature": list(X_test.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return frame


def plot_permutation_importance(frame: pd.DataFrame, path: Path) -> Path:
    """Horizontal bar chart of permutation importance (mean +/- std)."""
    ordered = frame.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(
        ordered["feature"],
        ordered["importance_mean"],
        xerr=ordered["importance_std"],
        color=TEAL,
        ecolor=SLATE,
        capsize=3,
    )
    ax.axvline(0, color=SLATE, linewidth=1)
    ax.set_xlabel("Mean drop in average precision when the feature is shuffled")
    ax.set_title("Permutation importance (held-out test set)",
                 fontweight="bold", color=NAVY)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_importance_series(series: pd.Series, title: str, path: Path) -> Path:
    """Generic importance bar chart (e.g. tree model impurity importances)."""
    ordered = series.sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(ordered.index, ordered.values, color=NAVY)
    ax.set_xlabel("Importance")
    ax.set_title(title, fontweight="bold", color=NAVY)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
