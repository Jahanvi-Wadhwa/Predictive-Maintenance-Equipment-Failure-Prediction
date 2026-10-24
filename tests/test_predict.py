"""Tests for input validation, model loading and prediction.

Model tests train a small pipeline on a real slice of the AI4I dataset (not
fabricated data) so they stay fast and independent of the trained artifact,
plus one test against the real saved model when it exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import MODEL_FEATURES
from src.features import make_model_pipeline, split_xy
from src.predict import (
    InputValidationError,
    ModelNotLoadedError,
    load_bundle,
    load_metrics,
    load_reference_stats,
    normalize_columns,
    predict_batch,
    predict_one,
    risk_category,
    validate_input,
)

RAW_ALIASES = pd.DataFrame({
    "Air temperature [K]": [298.1, 301.7],
    "Process temperature [K]": [308.6, 311.2],
    "Rotational speed [rpm]": [1551, 1420],
    "Torque [Nm]": [42.8, 55.1],
    "Tool wear [min]": [0, 120],
    "Type": ["M", "L"],
})


def valid_reading() -> dict:
    return {
        "air_temperature_k": 298.1,
        "process_temperature_k": 308.6,
        "rotational_speed_rpm": 1551.        ,
        "torque_nm": 42.8,
        "tool_wear_min": 0,
        "type": "M",
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_validate_accepts_valid_reading():
    frame = validate_input(valid_reading(), reference=None)
    assert list(frame.columns) == MODEL_FEATURES
    # Derived features are computed automatically.
    assert frame["power_kw"].iloc[0] == pytest.approx(
        42.8 * 1551 * 2 * 3.141592653589793 / 60 / 1000, rel=1e-6
    )
    assert frame["temp_diff_k"].iloc[0] == pytest.approx(308.6 - 298.1)


def test_validate_missing_columns_raises():
    with pytest.raises(InputValidationError, match="Missing required columns"):
        validate_input({"air_temperature_k": 300.0}, reference=None)


def test_validate_non_numeric_raises():
    reading = valid_reading() | {"torque_nm": "not a number"}
    with pytest.raises(InputValidationError, match="non-numeric"):
        validate_input(reading, reference=None)


def test_validate_invalid_product_type_raises():
    reading = valid_reading() | {"type": "XL"}
    with pytest.raises(InputValidationError, match="invalid values"):
        validate_input(reading, reference=None)


def test_validate_out_of_range_raises_with_reference():
    reference = {"torque_nm": {"min": 3.8, "max": 76.6, "median": 40.0}}
    reading = valid_reading() | {"torque_nm": 9999.0}
    with pytest.raises(InputValidationError, match="plausible"):
        validate_input(reading, reference=reference)


def test_validate_ignores_extra_columns():
    reading = valid_reading() | {"UDI": 123, "Machine failure": 0}
    frame = validate_input(reading, reference=None)
    assert list(frame.columns) == MODEL_FEATURES


def test_normalize_columns_maps_raw_ai4i_headers():
    mapped = normalize_columns(RAW_ALIASES)
    for canonical in ["air_temperature_k", "process_temperature_k",
                      "rotational_speed_rpm", "torque_nm", "tool_wear_min", "type"]:
        assert canonical in mapped.columns, f"{canonical} not mapped"
    frame = validate_input(mapped, reference=None)
    assert list(frame.columns) == MODEL_FEATURES


def test_normalize_columns_is_case_and_separator_tolerant():
    messy = pd.DataFrame({" Torque_Nm ": [40.0], "ROTATIONAL SPEED": [1500],
                          "air temp": [298.0], "process temp": [308.0],
                          "tool wear": [10], "product type": ["H"]})
    frame = validate_input(normalize_columns(messy), reference=None)
    assert frame["type"].iloc[0] == "H"
    assert frame["rotational_speed_rpm"].iloc[0] == 1500


def test_batch_validation_reports_all_bad_cells():
    batch = pd.DataFrame([valid_reading(), valid_reading() | {"torque_nm": 40.0}])
    batch.loc[1, "type"] = "Q"
    with pytest.raises(InputValidationError):
        validate_input(batch, reference=None)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def test_load_bundle_missing_file_raises(tmp_path):
    with pytest.raises(ModelNotLoadedError, match="Train it first"):
        load_bundle(tmp_path / "nope.joblib")


def test_load_bundle_corrupt_file_raises(tmp_path):
    bad = tmp_path / "corrupt.joblib"
    bad.write_bytes(b"this is not a joblib file")
    with pytest.raises(ModelNotLoadedError):
        load_bundle(bad)


def test_saved_model_loads_and_exposes_expected_fields(trained_bundle):
    assert trained_bundle["feature_names"] == MODEL_FEATURES
    thresholds = trained_bundle["risk_thresholds"]
    assert thresholds["moderate"] < thresholds["high"]
    assert trained_bundle["model_name"]


def test_reference_stats_and_metrics_load():
    stats = load_reference_stats()
    metrics = load_metrics()
    if stats is None or metrics is None:
        pytest.skip("artifacts not generated yet - run `python -m src.train`")
    for col in ["air_temperature_k", "torque_nm", "power_kw"]:
        assert stats[col]["min"] < stats[col]["max"]
    assert metrics["selected_model"] in metrics["models"]


# ---------------------------------------------------------------------------
# Prediction with a small real-data pipeline
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_bundle(small_raw):
    X, y = split_xy(small_raw)
    pipe = make_model_pipeline(LogisticRegression(max_iter=500, random_state=42))
    pipe.fit(X, y)
    return {
        "pipeline": pipe,
        "model_name": "tiny_logreg_test_model",
        "feature_names": MODEL_FEATURES,
        "risk_thresholds": {"moderate": 0.10, "high": 0.50},
    }


def test_predict_one_returns_probability_and_category(tiny_bundle):
    result = predict_one(valid_reading(), tiny_bundle)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["prediction"] in (0, 1)
    assert result["risk_category"] in ("Low", "Moderate", "High")
    assert result["model_name"] == "tiny_logreg_test_model"


def test_predict_batch_adds_prediction_columns(tiny_bundle):
    batch = normalize_columns(RAW_ALIASES)
    out = predict_batch(batch, tiny_bundle)
    assert len(out) == 2
    for col in ["failure_probability", "prediction", "risk_category"]:
        assert col in out.columns
    assert out["failure_probability"].between(0, 1).all()


def test_predict_batch_rejects_missing_column(tiny_bundle):
    broken = RAW_ALIASES.drop(columns=["Torque [Nm]"])
    with pytest.raises(InputValidationError, match="Missing required"):
        predict_batch(normalize_columns(broken), tiny_bundle)


def test_risk_category_boundaries():
    thresholds = {"moderate": 0.12, "high": 0.5}
    assert risk_category(0.01, thresholds) == "Low"
    assert risk_category(0.12, thresholds) == "Moderate"
    assert risk_category(0.49, thresholds) == "Moderate"
    assert risk_category(0.5, thresholds) == "High"
    assert risk_category(0.99, thresholds) == "High"


def test_real_trained_model_predicts_real_row(trained_bundle, raw_df):
    """End-to-end: the saved artifact scores a real dataset row."""
    sample = raw_df.iloc[0]
    reading = {
        "air_temperature_k": float(sample["Air temperature [K]"]),
        "process_temperature_k": float(sample["Process temperature [K]"]),
        "rotational_speed_rpm": float(sample["Rotational speed [rpm]"]),
        "torque_nm": float(sample["Torque [Nm]"]),
        "tool_wear_min": float(sample["Tool wear [min]"]),
        "type": str(sample["Type"]),
    }
    result = predict_one(reading, trained_bundle)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_category"] in ("Low", "Moderate", "High")