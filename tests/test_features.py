"""Tests for feature engineering and the preprocessing pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest

from src.config import FAILURE_MODE_COLUMNS, ID_COLUMNS, TARGET_COLUMN
from src.features import (
    add_engineered_features,
    assert_no_leakage,
    build_preprocessor,
    get_feature_names,
    make_model_pipeline,
    split_xy,
)


def test_engineered_features_values():
    df = pd.DataFrame({"torque_nm": [10.0], "rotational_speed_rpm": [600.0],
                       "process_temperature_k": [310.0], "air_temperature_k": [300.0]})
    out = add_engineered_features(df)
    # power = T[Nm] * w[rpm] * 2*pi / 60 / 1000  -> 10 * 600 * 2pi/60/1000 = 0.628 kW
    assert out["power_kw"].iloc[0] == pytest.approx(10 * 600 * 2 * np.pi / 60 / 1000)
    assert out["temp_diff_k"].iloc[0] == pytest.approx(10.0)
    # Original frame is not mutated.
    assert "power_kw" not in df.columns


def test_split_xy_excludes_leakage_columns(raw_df):
    X, y = split_xy(raw_df)
    forbidden = set(ID_COLUMNS + FAILURE_MODE_COLUMNS + [TARGET_COLUMN])
    assert not (set(X.columns) & forbidden), "leakage columns present in X"
    for col in ["air_temperature_k", "process_temperature_k", "rotational_speed_rpm",
                "torque_nm", "tool_wear_min", "power_kw", "temp_diff_k", "type"]:
        assert col in X.columns, f"missing expected feature {col}"
    assert len(X) == len(y) == len(raw_df)


def test_assert_no_leakage_raises():
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage(pd.DataFrame({"TWF": [0], "torque_nm": [1.0]}))


def test_preprocessor_output_shape_and_names():
    pre = build_preprocessor()
    df = pd.DataFrame({
        "air_temperature_k": [298.0, 301.0],
        "process_temperature_k": [308.0, 311.0],
        "rotational_speed_rpm": [1500.0, 1600.0],
        "torque_nm": [40.0, 50.0],
        "tool_wear_min": [10.0, 20.0],
        "power_kw": [6.0, 8.0],
        "temp_diff_k": [10.0, 10.0],
        "type": ["M", "L"],
    })
    transformed = pre.fit_transform(df)
    names = get_feature_names(pre)
    assert transformed.shape == (2, len(names))
    # 7 numeric features + one column per observed product type (L and M here).
    assert transformed.shape[1] == 7 + df["type"].nunique()
    assert "type_M" in names and "type_L" in names
    assert np.isfinite(transformed).all()


def test_preprocessor_handles_unknown_categories():
    pre = build_preprocessor()
    train = pd.DataFrame({
        "air_temperature_k": [298.0], "process_temperature_k": [308.0],
        "rotational_speed_rpm": [1500.0], "torque_nm": [40.0],
        "tool_wear_min": [10.0], "power_kw": [6.0], "temp_diff_k": [10.0],
        "type": ["M"],
    })
    pre.fit(train)
    unseen = train.assign(type="H")
    out = pre.transform(unseen)
    assert np.isfinite(out).all(), "unseen category must not break the transform"


def test_pipeline_step_order():
    pipe = make_model_pipeline("model_placeholder")
    assert list(pipe.named_steps) == ["preprocess", "model"]
    # With a resampler, ordering must be preprocess -> sampler -> model so
    # SMOTE can only ever see already-preprocessed training folds.
    with_sampler = make_model_pipeline("model_placeholder", sampler="sampler")
    assert list(with_sampler.named_steps) == ["preprocess", "sampler", "model"]


def test_engineered_features_do_not_mutate_input(raw_df):
    X, _ = split_xy(raw_df.head(50))
    original_cols = list(X.columns)
    add_engineered_features(X)
    assert list(X.columns) == original_cols