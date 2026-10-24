"""Model loading, input validation and prediction.

This module is the single prediction path shared by the Streamlit dashboard,
batch CSV scoring and the automated tests, so the dashboard never duplicates
training logic.

Run a quick self-check from the repository root after training::

    python -m src.predict
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    METRICS_PATH,
    MODEL_FEATURES,
    MODEL_PATH,
    NUMERIC_FEATURES,
    REFERENCE_STATS_PATH,
)

# Features derived inside validate_input rather than supplied by the user.
DERIVED_FEATURES = ["power_kw", "temp_diff_k"]
from src.features import add_engineered_features

# Fallback risk bands if no trained artifact exists yet. The real bands are
# derived from out-of-fold training operating points and stored with the model.
DEFAULT_THRESHOLDS = {"moderate": 0.10, "high": 0.50}

MODEL_ERROR_MESSAGE = (
    "Model artifact not found or unreadable. Train it first with: "
    "`python -m src.train`"
)


class ModelNotLoadedError(RuntimeError):
    """Raised when the saved model cannot be loaded."""


class InputValidationError(ValueError):
    """Raised when user input does not match the model's expected features."""


def load_bundle(model_path: Path | None = None) -> dict[str, Any]:
    """Load the saved model bundle.

    Args:
        model_path: Path to the joblib bundle. Defaults to
            ``models/failure_model.joblib``.

    Returns:
        Dict with keys ``pipeline``, ``model_name``, ``feature_names`` and
        ``risk_thresholds``.

    Raises:
        ModelNotLoadedError: If the file is missing or corrupt.
    """
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise ModelNotLoadedError(f"{MODEL_ERROR_MESSAGE}\nMissing file: {path}")
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        raise ModelNotLoadedError(
            f"{MODEL_ERROR_MESSAGE}\nCould not read {path}: {exc}"
        ) from exc

    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ModelNotLoadedError(
            f"{path} does not contain a valid model bundle. {MODEL_ERROR_MESSAGE}"
        )
    bundle.setdefault("feature_names", MODEL_FEATURES)
    bundle.setdefault("risk_thresholds", DEFAULT_THRESHOLDS)
    bundle.setdefault("model_name", "unknown")
    return bundle


def load_metrics(metrics_path: Path | None = None) -> dict[str, Any] | None:
    """Load metrics.json (returns ``None`` when training has not run yet)."""
    path = Path(metrics_path) if metrics_path else METRICS_PATH
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_reference_stats(path: Path | None = None) -> dict[str, Any] | None:
    """Load per-feature reference statistics used for form defaults/validation."""
    path = Path(path) if path else REFERENCE_STATS_PATH
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def validate_input(data: dict[str, Any] | pd.DataFrame,
                   reference: dict[str, Any] | None = None) -> pd.DataFrame:
    """Validate and normalize user-supplied readings into a model-ready frame.

    Accepts a dict (single reading) or a DataFrame (batch). Enforces that all
    model features are present, coerces numeric columns and checks each value
    against the observed dataset range (with a small tolerance) so obviously
    invalid readings are rejected instead of silently extrapolated.

    Raises:
        InputValidationError: With a list of all problems found.
    """
    frame = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data).copy()
    errors: list[str] = []

    # Derived features (power_kw, temp_diff_k) are computed from the raw
    # readings, so users only supply the original sensor columns.
    raw_numeric = [c for c in NUMERIC_FEATURES if c not in DERIVED_FEATURES]
    required = raw_numeric + CATEGORICAL_FEATURES

    missing = [c for c in required if c not in frame.columns]
    extra = [c for c in frame.columns if c not in MODEL_FEATURES]
    if missing:
        errors.append(f"Missing required columns: {missing}")
    if extra:
        # Extra columns are ignored rather than fatal, but the user should know.
        frame = frame[[c for c in frame.columns if c in MODEL_FEATURES]]

    if errors:
        raise InputValidationError("; ".join(errors))

    for col in raw_numeric:
        if col not in frame.columns:
            continue
        coerced = pd.to_numeric(frame[col], errors="coerce")
        if bool(coerced.isna().any()):
            errors.append(f"Column '{col}' contains non-numeric or missing values.")
        frame[col] = coerced

    for col in CATEGORICAL_FEATURES:
        frame[col] = frame[col].astype(str).str.strip().str.upper()
        allowed = {"L", "M", "H"}
        invalid = sorted(set(frame[col]) - allowed - {""})
        if invalid:
            errors.append(f"Column '{col}' has invalid values {invalid}; expected one of L, M, H.")

    if errors:
        raise InputValidationError(" | ".join(errors))

    # Engineered columns: compute if absent, keep user values if supplied.
    if not all(c in frame.columns for c in DERIVED_FEATURES):
        frame = add_engineered_features(frame)
        for col in DERIVED_FEATURES:
            if col not in frame.columns:
                frame[col] = np.nan

    reference = reference if reference is not None else load_reference_stats() or {}
    for col in NUMERIC_FEATURES:
        stats = reference.get(col)
        if not stats or col not in frame.columns:
            continue
        span = stats["max"] - stats["min"]
        tolerance = 0.25 * span if span > 0 else 1.0
        lo, hi = stats["min"] - tolerance, stats["max"] + tolerance
        out_of_range = frame[(frame[col] < lo) | (frame[col] > hi)]
        if not out_of_range.empty:
            errors.append(
                f"Column '{col}' has {len(out_of_range)} value(s) outside the plausible "
                f"range [{lo:.2f}, {hi:.2f}] observed in the AI4I dataset."
            )

    if errors:
        raise InputValidationError(" | ".join(errors))

    return frame[MODEL_FEATURES]


# Accepted aliases -> canonical model feature name. Lets users upload the raw
# AI4I CSV or a spreadsheet with human-readable headers.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "air_temperature_k": (
        "air_temperature_k", "air temperature [k]", "air temperature", "air temp",
    ),
    "process_temperature_k": (
        "process_temperature_k", "process temperature [k]", "process temperature",
        "process temp",
    ),
    "rotational_speed_rpm": (
        "rotational_speed_rpm", "rotational speed [rpm]", "rotational speed",
        "speed", "rpm",
    ),
    "torque_nm": ("torque_nm", "torque [nm]", "torque"),
    "tool_wear_min": ("tool_wear_min", "tool wear [min]", "tool wear", "wear"),
    "type": ("type", "product type", "product quality"),
}


def _normalize_label(label: str) -> str:
    return " ".join(str(label).strip().lower().replace("_", " ").split())


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Map common column-name variants onto the canonical model features.

    Handles the raw AI4I headers (``Air temperature [K]``), snake_case names
    with different separators and capitalisation, e.g. an exported
    spreadsheet. Unknown columns are left untouched (they are ignored later
    by :func:`validate_input`).
    """
    lookup = {
        _normalize_label(alias): canonical
        for canonical, aliases in COLUMN_ALIASES.items()
        for alias in aliases
    }
    rename = {
        col: lookup[_normalize_label(col)]
        for col in data.columns
        if _normalize_label(col) in lookup
    }
    return data.rename(columns=rename)


def risk_category(probability: float, thresholds: dict[str, float] | None = None) -> str:
    """Map a failure probability to Low / Moderate / High relative risk."""
    thresholds = thresholds or DEFAULT_THRESHOLDS
    moderate = float(thresholds.get("moderate", DEFAULT_THRESHOLDS["moderate"]))
    high = float(thresholds.get("high", DEFAULT_THRESHOLDS["high"]))
    if probability >= high:
        return "High"
    if probability >= moderate:
        return "Moderate"
    return "Low"


def predict_one(
    reading: dict[str, Any], bundle: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Predict failure risk for a single machine reading.

    Args:
        reading: Dict with the seven numeric features and ``type``.
        bundle: Loaded model bundle; loaded on demand if omitted.

    Returns:
        Dict with ``probability``, ``prediction`` (0/1 at threshold 0.5),
        ``risk_category``, ``model_name`` and the thresholds used.
    """
    bundle = bundle if bundle is not None else load_bundle()
    frame = validate_input(reading)
    pipeline = bundle["pipeline"]
    probability = float(pipeline.predict_proba(frame)[0, 1])
    thresholds = bundle.get("risk_thresholds") or DEFAULT_THRESHOLDS
    return {
        "probability": probability,
        "prediction": int(probability >= 0.5),
        "risk_category": risk_category(probability, thresholds),
        "model_name": bundle.get("model_name", "unknown"),
        "thresholds": thresholds,
    }


def predict_batch(
    data: pd.DataFrame, bundle: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Score a batch of readings and return the input plus prediction columns.

    Returns:
        The original frame augmented with ``failure_probability``,
        ``prediction`` and ``risk_category``.

    Raises:
        InputValidationError: If required columns are missing/invalid.
    """
    bundle = bundle if bundle is not None else load_bundle()
    validated = validate_input(data)
    pipeline = bundle["pipeline"]
    probabilities = pipeline.predict_proba(validated)[:, 1]
    thresholds = bundle.get("risk_thresholds") or DEFAULT_THRESHOLDS

    out = data.copy().reset_index(drop=True)
    out["failure_probability"] = np.round(probabilities, 4)
    out["prediction"] = (probabilities >= 0.5).astype(int)
    out["risk_category"] = [
        risk_category(p, thresholds) for p in probabilities
    ]
    return out


def _self_check() -> int:
    """Score one real row from the dataset to verify the saved artifact."""
    try:
        bundle = load_bundle()
    except ModelNotLoadedError as exc:
        print(f"SKIP: {exc}")
        return 0
    from src.data import load_dataset

    raw = load_dataset()
    sample = raw.iloc[0]
    reading = {
        "air_temperature_k": float(sample["Air temperature [K]"]),
        "process_temperature_k": float(sample["Process temperature [K]"]),
        "rotational_speed_rpm": float(sample["Rotational speed [rpm]"]),
        "torque_nm": float(sample["Torque [Nm]"]),
        "tool_wear_min": float(sample["Tool wear [min]"]),
        "type": str(sample["Type"]),
    }
    # Derived features are added inside validate_input, exactly as training does.
    result = predict_one(reading, bundle)
    print(
        f"Model: {result['model_name']} | probability={result['probability']:.4f} "
        f"| category={result['risk_category']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_self_check())