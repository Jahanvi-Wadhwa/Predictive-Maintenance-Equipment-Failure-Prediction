"""Exploratory data analysis (EDA).

Produces a markdown EDA report and PNG charts from the *real* dataset.

Run from the repository root::

    python -m src.eda

Outputs:
    reports/eda_report.md
    reports/figures/eda_*.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    FAILURE_MODE_COLUMNS,
    FIGURES_DIR,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
)
from src.data import load_dataset
from src.features import load_raw_feature_frame

PALETTE = {
    "navy": "#0f2a4a",
    "slate": "#475569",
    "teal": "#0d9488",
    "white": "#ffffff",
    "light": "#e2e8f0",
    "alert": "#d97706",
}


def check_quality(df: pd.DataFrame) -> dict:
    """Missing-value, duplicate and constant-column checks."""
    duplicates = int(df.duplicated().sum())
    missing = {c: int(v) for c, v in df.isna().sum().items() if v > 0}
    id_dupe = int(df.duplicated(subset=["UDI"]).sum()) if "UDI" in df.columns else None
    constants = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "duplicate_rows": duplicates,
        "duplicate_udi": id_dupe,
        "missing_values": missing or {"(none)": 0},
        "constant_columns": constants,
    }


def class_distribution(df: pd.DataFrame) -> dict:
    """Return class counts and the failure rate (imbalance analysis)."""
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    n = int(counts.sum())
    failures = int(counts.get(1, 0))
    return {
        "n_samples": n,
        "n_failures": failures,
        "n_non_failures": int(counts.get(0, 0)),
        "failure_rate": failures / n if n else 0.0,
        "imbalance_ratio": (failures and (n - failures) / failures) or float("nan"),
    }


def failure_mode_breakdown(df: pd.DataFrame) -> pd.Series:
    """Counts per failure mode among rows flagged as failed."""
    failed = df[df[TARGET_COLUMN] == 1]
    return failed[FAILURE_MODE_COLUMNS].sum().astype(int).sort_values(ascending=False)


def plot_class_distribution(df: pd.DataFrame, path: Path) -> Path:
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["No failure (0)", "Failure (1)"], counts.values,
                  color=[PALETTE["teal"], PALETTE["alert"]], width=0.55)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("Number of machines")
    ax.set_title("Class distribution: machine failure", fontweight="bold", color=PALETTE["navy"])
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_feature_distributions(X: pd.DataFrame, y: pd.Series, path: Path) -> Path:
    numeric_cols = [c for c in X.columns if X[c].dtype.kind in "fi" and c != "type"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), numeric_cols):
        ax.hist(X.loc[y == 0, col].dropna(), bins=40, alpha=0.65,
                color=PALETTE["teal"], label="No failure")
        ax.hist(X.loc[y == 1, col].dropna(), bins=40, alpha=0.75,
                color=PALETTE["alert"], label="Failure")
        ax.set_title(col, fontsize=10, color=PALETTE["navy"])
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[len(numeric_cols):]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Feature distributions by failure status",
                 fontweight="bold", color=PALETTE["navy"])
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_correlation(X: pd.DataFrame, y: pd.Series, path: Path) -> Path:
    data = X.select_dtypes(include=[np.number]).copy()
    data[TARGET_COLUMN] = y.values
    corr = data.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.columns)), corr.columns, fontsize=8)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            value = corr.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(value) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Pearson correlation (features + target)", fontweight="bold",
                 color=PALETTE["navy"])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_failure_modes(df: pd.DataFrame, path: Path) -> Path:
    modes = failure_mode_breakdown(df)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(modes.index, modes.values, color=PALETTE["navy"], width=0.6)
    for i, v in enumerate(modes.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Occurrences among failed machines")
    ax.set_title("Failure modes among failed machines (multi-label)",
                 fontweight="bold", color=PALETTE["navy"])
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_eda(write_report: bool = True) -> dict:
    """Run the full EDA and save figures + markdown report.

    Returns:
        A dictionary with quality checks, class distribution and figure paths.
    """
    df = load_dataset()
    quality = check_quality(df)
    dist = class_distribution(df)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "class_distribution": plot_class_distribution(df, FIGURES_DIR / "eda_class_distribution.png"),
        "feature_distributions": plot_feature_distributions(
            load_raw_feature_frame(df), df[TARGET_COLUMN].astype(int),
            FIGURES_DIR / "eda_feature_distributions.png"),
        "correlation": plot_correlation(load_raw_feature_frame(df),
                                        df[TARGET_COLUMN].astype(int),
                                        FIGURES_DIR / "eda_correlation.png"),
        "failure_modes": plot_failure_modes(df, FIGURES_DIR / "eda_failure_modes.png"),
    }

    if write_report:
        _write_report(df, quality, dist, figures)

    return {"quality": quality, "class_distribution": dist,
            "figures": {k: str(v) for k, v in figures.items()}}


def _write_report(df: pd.DataFrame, quality: dict, dist: dict,
                  figures: dict[str, Path]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    modes = failure_mode_breakdown(df)
    desc = df.describe().T.round(3)
    lines = [
        "# Exploratory Data Analysis - AI4I 2020 Predictive Maintenance",
        "",
        "Generated by `python -m src.eda` from the real dataset "
        f"(`{RAW_DATA_PATH.name}`).",
        "",
        "## 1. Quality checks",
        "",
        f"- Rows x columns: **{quality['n_rows']} x {quality['n_cols']}**",
        f"- Duplicate rows: **{quality['duplicate_rows']}**",
        f"- Duplicate `UDI` ids: **{quality['duplicate_udi']}**",
        f"- Columns with missing values: "
        + (", ".join(f"`{k}` ({v})" for k, v in quality["missing_values"].items())
           if list(quality["missing_values"]) != ["(none)"] else "**none**"),
        f"- Constant columns: "
        + (", ".join(quality["constant_columns"]) if quality["constant_columns"] else "**none**"),
        "",
        "## 2. Class distribution (imbalance)",
        "",
        f"- Samples: {dist['n_samples']}",
        f"- Failures: {dist['n_failures']} ({dist['failure_rate'] * 100:.2f}%)",
        f"- Non-failures: {dist['n_non_failures']}",
        f"- Imbalance ratio (non-failure : failure) = **{dist['imbalance_ratio']:.1f} : 1**",
        "",
        "The target is heavily imbalanced, so accuracy alone is misleading. "
        "The pipeline therefore reports precision, recall, F1, ROC-AUC and "
        "average precision (PR-AUC), and class imbalance is handled explicitly.",
        "",
        "## 3. Leakage review",
        "",
        "- `TWF`, `HDF`, `PWF`, `OSF`, `RNF` are the *components* of "
        "`Machine failure` (failure = any of them). They are excluded from "
        "model inputs and used only for analysis.",
        "- `UDI` (row id) and `Product ID` (product code) are identifiers and "
        "are excluded.",
        "",
        "## 4. Failure modes among failed machines",
        "",
        "| Failure mode | Occurrences |",
        "|---|---|",
    ]
    lines += [f"| {k} | {int(v)} |" for k, v in modes.items()]
    lines += [
        "",
        "> Note: a machine can have more than one failure mode active, so counts "
        "sum to more than the number of failed machines.",
        "",
        "## 5. Numeric feature summary",
        "",
        desc.to_markdown(),
        "",
        "## 6. Figures",
        "",
    ]
    lines += [f"- `{Path(p).name}`" for p in figures.values()]
    report_path = REPORTS_DIR / "eda_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"EDA report written to {report_path}")
    return report_path


if __name__ == "__main__":
    result = run_eda()
    print(f"Failure rate: {result['class_distribution']['failure_rate']:.4f}")
    print(f"Figures saved in {FIGURES_DIR}")
