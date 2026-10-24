"""Smoke tests for the training pipeline plus artifact consistency checks.

These keep the training logic honest without running the full 2-minute
training script inside pytest.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from src.config import METRICS_PATH, MODEL_PATH, RANDOM_SEED, RECALL_TARGET, PRECISION_TARGET
from src.features import split_xy
from src.train import (
    build_pipelines,
    define_models,
    run_cross_validation,
    select_model,
    split_data,
)


def test_define_models_contains_three_strategies():
    models = define_models()
    assert set(models) == {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
    }
    # Exactly one model uses in-pipeline resampling; the others use class weights.
    assert models["logistic_regression"]["sampler"] is not None
    assert models["random_forest"]["sampler"] is None
    assert models["hist_gradient_boosting"]["sampler"] is None
    for cfg in models.values():
        assert cfg["description"]


def test_stratified_split_preserves_both_classes(small_raw):
    X, y = split_xy(small_raw)
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}
    assert len(X_train) + len(X_test) == len(X)
    # Stratification keeps the positive rate close in both halves.
    assert abs(y_train.mean() - y_test.mean()) < 0.05
    # Deterministic given RANDOM_SEED.
    _, _, y_train2, _ = split_data(X, y)
    assert y_train.equals(y_train2)


def test_cross_validation_returns_finite_scores(small_raw):
    """Tiny CV run (logistic regression only, 2 folds) on a real data slice."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate

    X, y = split_xy(small_raw)
    pipe = build_pipelines()["logistic_regression"]
    scores = cross_validate(
        pipe,
        X,
        y,
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED),
        scoring={"roc_auc": "roc_auc", "average_precision": "average_precision"},
    )
    for key in ("test_roc_auc", "test_average_precision"):
        assert np.isfinite(scores[key]).all(), f"{key} contains non-finite values"
        assert 0.0 <= scores[key].mean() <= 1.0


def test_select_model_is_pure_cv_logic():
    cv_results = {
        "a": {"average_precision_mean": 0.50},
        "b": {"average_precision_mean": 0.90},
        "c": {"average_precision_mean": 0.70},
    }
    assert select_model(cv_results) == "b"


def test_run_cross_validation_returns_all_metrics(small_raw):
    X, y = split_xy(small_raw)
    pipelines = {"logistic_regression": build_pipelines()["logistic_regression"]}
    from src.config import CV_FOLDS

    # Use 2 folds for speed on the small slice.
    import src.train as train_mod

    original = train_mod.CV_FOLDS
    train_mod.CV_FOLDS = 2
    try:
        results = run_cross_validation(pipelines, X, y)
    finally:
        train_mod.CV_FOLDS = original
    cv = results["logistic_regression"]
    for metric in ("roc_auc", "average_precision", "recall", "f1"):
        assert f"{metric}_mean" in cv and f"{metric}_std" in cv
        assert math.isfinite(cv[f"{metric}_mean"])


# ---------------------------------------------------------------------------
# Artifact consistency (skipped until `python -m src.train` has been run)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not METRICS_PATH.exists(), reason="metrics.json missing")
def test_metrics_json_is_internally_consistent():
    payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    assert payload["selected_model"] in payload["models"]
    assert payload["cv_folds"] >= 2
    assert payload["risk_thresholds"]["moderate"] < payload["risk_thresholds"]["high"]
    assert payload["risk_thresholds"]["high"] <= 1.0

    dataset = payload["dataset"]
    assert dataset["n_failures"] / dataset["n_rows"] == pytest.approx(
        dataset["failure_rate"], rel=1e-9
    )

    for name, res in payload["models"].items():
        m = res["test"]["metrics"]
        tn, fp, fn, tp = m["confusion_matrix"][0][0], m["confusion_matrix"][0][1], \
            m["confusion_matrix"][1][0], m["confusion_matrix"][1][1]
        assert tn + fp + fn + tp == m["n_samples"] == payload["split"]["n_test"]
        # Precision / recall / F1 must match the confusion matrix exactly.
        assert m["precision"] == pytest.approx(tp / (tp + fp), abs=1e-9)
        assert m["recall"] == pytest.approx(tp / (tp + fn), abs=1e-9)
        assert m["f1"] == pytest.approx(
            2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"]), abs=1e-9
        )
        for key in ("roc_auc", "average_precision"):
            assert m[key] is None or 0.0 <= m[key] <= 1.0
        # CV scores also bounded
        for key, value in res["cv"].items():
            assert math.isfinite(value), f"{name}.{key} not finite"

    # Total failures across the split must equal dataset total.
    assert payload["split"]["train_failures"] + payload["split"]["test_failures"] == \
        dataset["n_failures"]

    # Risk band rule constants are recorded honestly.
    assert payload["risk_threshold_stats"]["moderate"]["oof_precision"] >= PRECISION_TARGET - 0.05
    assert payload["risk_threshold_stats"]["moderate"]["oof_recall"] >= RECALL_TARGET - 0.05


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model artifact missing")
def test_saved_model_matches_metrics():
    from src.predict import load_bundle

    payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    bundle = load_bundle()
    assert bundle["model_name"] == payload["selected_model"]