"""End-to-end training pipeline for predictive maintenance.

What this script does, in order:

1. Loads the AI4I 2020 dataset (downloads it if missing).
2. Runs leakage checks and a stratified 80/20 train/test split.
   The test set is touched **only** for final evaluation, never for model
   selection or threshold tuning.
3. Trains three classifiers with preprocessing (and, where used, SMOTE)
   *inside* the pipeline, so resampling/imputation/scaling happen only on
   training folds:

   - Logistic Regression  (interpretable baseline, SMOTE + class weights)
   - Random Forest         (bagged trees, balanced class weights)
   - HistGradientBoosting  (gradient boosting, balanced class weights)

4. Compares them with 5-fold stratified cross-validation on the training set
   (ROC-AUC, average precision / PR-AUC, recall, F1).
5. Evaluates all three on the untouched test set and selects the winner using
   the CV average precision (PR-AUC) only.
6. Derives risk-band thresholds from out-of-fold training probabilities.
7. Computes permutation importance for the selected model.
8. Saves the fitted pipeline plus every artifact the dashboard reads:

   - models/failure_model.joblib
   - models/artifacts/metrics.json
   - models/artifacts/permutation_importance.csv
   - models/artifacts/reference_stats.json
   - reports/figures/*.png

Run from the repository root::

    python -m src.train
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

from src.config import (
    ARTIFACTS_DIR,
    CV_FOLDS,
    CV_SCORING,
    HIGH_RISK_THRESHOLD,
    METRICS_PATH,
    MODEL_PATH,
    MODEL_FEATURES,
    PERM_IMPORTANCE_PATH,
    PRECISION_TARGET,
    RANDOM_SEED,
    RECALL_TARGET,
    REFERENCE_STATS_PATH,
    TEST_SIZE,
)
from src.data import load_dataset
from src.evaluate import (
    compute_classification_metrics,
    curve_data,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_pr_curves,
    plot_roc_curves,
)
from src.explain import compute_permutation_importance, plot_permutation_importance
from src.features import assert_no_leakage, make_model_pipeline, split_xy

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
SELECTION_METRIC = "average_precision_mean"


def define_models() -> dict[str, dict[str, Any]]:
    """Return the three candidate models and their imbalance strategy.

    Each entry holds:

    * ``model``       - unfitted sklearn classifier
    * ``sampler``     - optional imblearn resampler (used only in the pipeline,
                        i.e. only on training folds)
    * ``description`` - human-readable summary for the README/dashboard
    """
    return {
        "logistic_regression": {
            "model": LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
            # SMOTE makes the minority class visible to the linear boundary.
            "sampler": SMOTE(random_state=RANDOM_SEED),
            "description": (
                "Linear baseline. Standardized features, SMOTE resampling "
                "inside the pipeline; coefficients are easy to interpret."
            ),
        },
        "random_forest": {
            "model": RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_SEED,
            ),
            "sampler": None,
            "description": (
                "Bagged decision trees with balanced class weights; robust to "
                "non-linear effects and feature scaling."
            ),
        },
        "hist_gradient_boosting": {
            "model": HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
            "sampler": None,
            "description": (
                "Gradient boosting on decision trees with balanced class "
                "weights; usually strongest on tabular data."
            ),
        },
    }


def build_pipelines() -> dict[str, Any]:
    """Instantiate the pipelines (preprocessing -> optional SMOTE -> model)."""
    return {
        name: make_model_pipeline(cfg["model"], cfg["sampler"])
        for name, cfg in define_models().items()
    }


# ---------------------------------------------------------------------------
# Training steps
# ---------------------------------------------------------------------------
def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Stratified train/test split with a fixed seed."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )


def run_cross_validation(pipelines: dict, X_train: pd.DataFrame,
                         y_train: pd.Series) -> dict[str, dict[str, float]]:
    """5-fold stratified CV on the training set only."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results: dict[str, dict[str, float]] = {}
    for name, pipeline in pipelines.items():
        print(f"  Cross-validating {name} ...", flush=True)
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=CV_SCORING,
            n_jobs=1,
            error_score="raise",
        )
        results[name] = {
            f"{metric}_mean": float(np.mean(values))
            for metric, values in (
                (m, scores[f"test_{m}"]) for m in CV_SCORING
            )
        } | {
            f"{metric}_std": float(np.std(values))
            for metric, values in (
                (m, scores[f"test_{m}"]) for m in CV_SCORING
            )
        }
    return results


def select_model(cv_results: dict[str, dict[str, float]]) -> str:
    """Pick the model with the best mean CV average precision (PR-AUC).

    Average precision is the selection criterion because failures are rare;
    it weights precision across recall levels and is not inflated by the large
    number of true negatives the way accuracy is. Only CV scores are used -
    the test set never influences selection.
    """
    return max(cv_results, key=lambda name: cv_results[name][SELECTION_METRIC])


def evaluate_on_test(pipelines: dict, X_test: pd.DataFrame,
                     y_test: pd.Series) -> dict[str, dict[str, Any]]:
    """Compute metrics + ROC/PR curve data for every fitted pipeline."""
    results: dict[str, dict[str, Any]] = {}
    for name, pipeline in pipelines.items():
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        results[name] = {
            "metrics": compute_classification_metrics(
                y_test.to_numpy(), y_pred, y_proba
            ),
            "curves": curve_data(y_test.to_numpy(), y_proba),
        }
    return results


def derive_risk_thresholds(
    pipeline, X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[dict[str, float], dict[str, Any]]:
    """Derive Low/Moderate/High risk bands from out-of-fold training scores.

    Out-of-fold predictions avoid the optimistic bias of in-sample scores and
    come from CV on the training set only - the test set is never used.

    Rules (documented, reproducible, based on real OOF operating points):

    * ``moderate`` - the *lowest* probability cut-off, scanned on a 0.01 grid,
      at which out-of-fold flags reach ``PRECISION_TARGET`` precision while
      keeping at least ``RECALL_TARGET`` recall. If no grid point satisfies
      both, the F1-optimal threshold is used instead.
    * ``high``     - ``HIGH_RISK_THRESHOLD`` (0.5), the model's own default
      classification threshold.

    Bands are *relative* to how the model scores unseen training data; they
    are not validated industrial safety limits.

    Returns:
        (thresholds, stats) where ``stats`` records the realized precision,
        recall and flag count at each cut-off, stored in metrics.json.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]

    def _pr_at(threshold: float) -> tuple[float, float, int]:
        flags = (oof >= threshold).astype(int)
        n_flagged = int(flags.sum())
        if n_flagged == 0:
            return 0.0, 0.0, 0
        return (
            float(precision_score(y_train, flags, zero_division=0)),
            float(recall_score(y_train, flags, zero_division=0)),
            n_flagged,
        )

    moderate = None
    for step in range(1, 99):
        candidate = step / 100
        precision, recall, _ = _pr_at(candidate)
        if precision >= PRECISION_TARGET and recall >= RECALL_TARGET:
            moderate = candidate
            break
    if moderate is None:
        # Fallback: F1-optimal operating point on OOF predictions.
        grid = np.arange(0.01, 0.99, 0.01)
        f1s = [
            float(f1_score(y_train, (oof >= candidate).astype(int), zero_division=0))
            for candidate in grid
        ]
        moderate = float(grid[int(np.argmax(f1s))])

    thresholds = {"moderate": float(moderate), "high": float(HIGH_RISK_THRESHOLD)}
    stats = {}
    for band, threshold in thresholds.items():
        precision, recall, n_flagged = _pr_at(threshold)
        stats[band] = {
            "threshold": threshold,
            "oof_precision": round(precision, 4),
            "oof_recall": round(recall, 4),
            "oof_flagged": n_flagged,
        }
    print(f"  Risk band cut-offs: {thresholds} | OOF stats: {stats}")
    return thresholds, stats


def compute_reference_stats(X: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Min/max/median/mean per model feature -> dashboard form defaults & bounds."""
    stats: dict[str, dict[str, float]] = {}
    for col in MODEL_FEATURES:
        if col not in X.columns:
            continue
        series = X[col]
        if series.dtype.kind in "fi":
            stats[col] = {
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
                "mean": float(series.mean()),
                "p05": float(series.quantile(0.05)),
                "p95": float(series.quantile(0.95)),
            }
        else:
            stats[col] = {"levels": sorted(str(v) for v in series.dropna().unique())}
    return stats


def native_feature_importance(pipeline) -> pd.Series | None:
    """Tree/linear model's own feature importances, if available."""
    model = pipeline.named_steps["model"]
    pre = pipeline.named_steps["preprocess"]
    names = list(pre.get_feature_names_out())
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_).ravel())
    else:
        return None
    if len(values) != len(names):
        return None
    return pd.Series(values, index=names, name="importance")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def train_all(save: bool = True, make_plots: bool = True) -> dict[str, Any]:
    """Run the complete training workflow and (optionally) save all artifacts.

    Returns:
        The ``metrics.json`` payload so callers (tests, notebooks) can inspect
        real results without re-reading files.
    """
    print("Loading dataset ...")
    df = load_dataset()
    X, y = split_xy(df)
    assert_no_leakage(X)

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(
        f"Train: {len(X_train)} rows ({int(y_train.sum())} failures) | "
        f"Test: {len(X_test)} rows ({int(y_test.sum())} failures)"
    )

    print("Building pipelines and running 5-fold cross-validation ...")
    pipelines = build_pipelines()
    cv_results = run_cross_validation(pipelines, X_train, y_train)

    print("Fitting every model on the full training set ...")
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)

    print("Evaluating on the untouched test set ...")
    test_results = evaluate_on_test(pipelines, X_test, y_test)

    selected = select_model(cv_results)
    print(f"Selected model (best CV {SELECTION_METRIC}): {selected}")

    best_pipeline = pipelines[selected]
    thresholds, threshold_stats = derive_risk_thresholds(best_pipeline, X_train, y_train)
    print(f"Risk thresholds: {thresholds}")

    print("Computing permutation importance on the test set ...")
    perm = compute_permutation_importance(best_pipeline, X_test, y_test)
    native = native_feature_importance(best_pipeline)

    split_info = {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_SEED,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_failures": int(y_train.sum()),
        "test_failures": int(y_test.sum()),
    }
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "AI4I 2020 Predictive Maintenance Dataset",
            "source": "UCI Machine Learning Repository (dataset id 601)",
            "url": "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset",
            "n_rows": int(len(df)),
            "failure_rate": float(y.mean()),
            "n_failures": int(y.sum()),
            "features": MODEL_FEATURES,
        },
        "split": split_info,
        "cv_folds": CV_FOLDS,
        "selection_criterion": (
            "Highest mean 5-fold CV average precision (PR-AUC) on the "
            "training set; the test set was not used for selection."
        ),
        "selected_model": selected,
        "risk_thresholds": thresholds,
        "risk_threshold_stats": threshold_stats,
        "risk_band_rules": {
            "moderate": (
                f"Lowest cut-off with OOF precision >= {PRECISION_TARGET} "
                f"and recall >= {RECALL_TARGET} on the training set (5-fold OOF)."
            ),
            "high": (
                f"{HIGH_RISK_THRESHOLD} - the model's default classification "
                "threshold. Project-defined bands, not validated industrial limits."
            ),
        },
        "models": {
            name: {
                "description": define_models()[name]["description"],
                "uses_smote": define_models()[name]["sampler"] is not None,
                "cv": cv_results[name],
                "test": test_results[name],
            }
            for name in pipelines
        },
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    if make_plots:
        _make_plots(payload, pipelines, cv_results, test_results, perm)

    if save:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": best_pipeline,
                "model_name": selected,
                "feature_names": MODEL_FEATURES,
                "risk_thresholds": thresholds,
                "trained_at": payload["generated_at"],
            },
            MODEL_PATH,
        )
        METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        perm.to_csv(PERM_IMPORTANCE_PATH, index=False)
        if native is not None:
            native.rename("importance").to_csv(ARTIFACTS_DIR / "native_importance.csv")
        REFERENCE_STATS_PATH.write_text(
            json.dumps(compute_reference_stats(X), indent=2), encoding="utf-8"
        )
        print(f"Saved model      -> {MODEL_PATH}")
        print(f"Saved metrics    -> {METRICS_PATH}")
        print(f"Saved importance -> {PERM_IMPORTANCE_PATH}")
        print(f"Saved ref stats  -> {REFERENCE_STATS_PATH}")

    return payload


def _make_plots(payload, pipelines, cv_results, test_results, perm) -> None:
    from src.config import FIGURES_DIR

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    curves_roc, curves_pr = {}, {}
    for name, res in test_results.items():
        label = name.replace("_", " ").title()
        plot_confusion_matrix(
            res["metrics"]["confusion_matrix"],
            f"Confusion matrix - {label} (test set)",
            FIGURES_DIR / f"confusion_matrix_{name}.png",
        )
        curves_roc[label] = res["curves"]
        curves_pr[label] = res["curves"]
    plot_roc_curves(curves_roc, FIGURES_DIR / "roc_curves.png")
    plot_pr_curves(
        curves_pr, payload["dataset"]["failure_rate"], FIGURES_DIR / "pr_curves.png"
    )
    plot_model_comparison(cv_results, FIGURES_DIR / "model_comparison.png")
    plot_permutation_importance(perm, FIGURES_DIR / "permutation_importance.png")


def print_summary(payload: dict[str, Any]) -> None:
    """Print a compact table of CV and test metrics for the terminal."""
    print("\n=== Cross-validation (training set, 5-fold) ===")
    header = f"{'model':24} {'ROC-AUC':>9} {'PR-AUC':>9} {'recall':>9} {'f1':>9}"
    print(header)
    for name, res in payload["models"].items():
        cv = res["cv"]
        print(
            f"{name:24} {cv['roc_auc_mean']:.4f}     {cv['average_precision_mean']:.4f}     "
            f"{cv['recall_mean']:.4f}     {cv['f1_mean']:.4f}"
        )
    print("\n=== Held-out test set ===")
    print(header)
    for name, res in payload["models"].items():
        m = res["test"]["metrics"]
        print(
            f"{name:24} {m['roc_auc']:.4f}     {m['average_precision']:.4f}     "
            f"{m['recall']:.4f}     {m['f1']:.4f}"
        )
    print(f"\nSelected model: {payload['selected_model']}")
    for name, res in payload["models"].items():
        m = res["test"]["metrics"]
        tn, fp, fn, tp = np.asarray(m["confusion_matrix"]).ravel()
        print(
            f"  {name}: precision={m['precision']:.4f} "
            f"TN={tn} FP={fp} FN={fn} TP={tp}"
        )


if __name__ == "__main__":
    try:
        metrics = train_all()
    except Exception as exc:  # noqa: BLE001
        print(f"TRAINING FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    print_summary(metrics)