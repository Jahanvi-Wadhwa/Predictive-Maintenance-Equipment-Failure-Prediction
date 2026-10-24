"""Feature definitions, engineering and the sklearn preprocessing pipeline.

Design decisions (documented in README):

* ``UDI`` / ``Product ID`` are identifiers -> dropped.
* ``TWF``, ``HDF``, ``PWF``, ``OSF``, ``RNF`` are failure-mode labels that
  *compose* the target ``Machine failure`` -> dropped from model inputs to
  prevent target leakage.
* Two features are derived from raw sensor readings only (no extra data, no
  leakage): a mechanical power proxy (torque x speed) and the process/air
  temperature gap.
* Preprocessing = median imputation (robust to the rare missing value),
  standard scaling for numeric features and one-hot encoding for the
  categorical product ``Type``. Everything lives inside the sklearn
  :class:`~sklearn.pipeline.Pipeline`, so it is fit only on training folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    FAILURE_MODE_COLUMNS,
    ID_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features computed only from raw sensor readings.

    New columns:

    * ``power_kw``   - mechanical power proxy: ``torque * speed`` scaled to kW
      (``T [Nm] * w [rpm] * 2*pi / 60 / 1000``).
    * ``temp_diff_k`` - ``process_temperature - air_temperature``; the heating
      caused by the process relative to ambient air.

    Both are pure transformations of existing inputs, so they introduce no
    leakage and no external data requirements.
    """
    out = df.copy()
    if {"torque_nm", "rotational_speed_rpm"} <= set(out.columns):
        out["power_kw"] = out["torque_nm"] * out["rotational_speed_rpm"] * 2 * np.pi / 60 / 1000
    if {"process_temperature_k", "air_temperature_k"} <= set(out.columns):
        out["temp_diff_k"] = out["process_temperature_k"] - out["air_temperature_k"]
    return out


def load_raw_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw dataset columns to clean names and add engineered features.

    Args:
        df: The raw dataset returned by :func:`src.data.load_dataset`.

    Returns:
        A frame with clean numeric/categorical feature names (no target, no
        identifiers, no failure-mode columns).
    """
    from src.config import RAW_TO_CLEAN  # local import to avoid cycles

    renamed = df.rename(columns=RAW_TO_CLEAN)
    keep = NUMERIC_FEATURES[:5] + CATEGORICAL_FEATURES  # engineered added next
    available = [c for c in keep if c in renamed.columns]
    features = renamed[available].copy()
    features = add_engineered_features(features)
    return features


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the raw dataset into a leakage-free feature frame and the target."""
    X = load_raw_feature_frame(df)
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def assert_no_leakage(df: pd.DataFrame) -> None:
    """Fail fast if a failure-mode or identifier column sneaks into the features."""
    forbidden = set(ID_COLUMNS + FAILURE_MODE_COLUMNS) & set(df.columns)
    if forbidden:
        raise ValueError(
            f"Target leakage detected: these columns must not be model inputs: {sorted(forbidden)}"
        )


def build_preprocessor() -> ColumnTransformer:
    """Build the preprocessing column transformer shared by all models.

    Numeric columns -> median imputation + standard scaling.
    Categorical ``type`` -> most-frequent imputation + one-hot encoding with
    ``handle_unknown='ignore'`` so unseen product types at prediction time do
    not crash the pipeline.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return the feature names produced by :func:`build_preprocessor`."""
    return list(preprocessor.get_feature_names_out())


def make_model_pipeline(model, sampler=None) -> ImbPipeline:
    """Assemble an imblearn pipeline: preprocessing -> (optional resampling) -> model.

    Because resampling sits *inside* the pipeline, SMOTE is applied only to
    training folds during cross-validation and never sees the test set.

    Args:
        model: Any sklearn-compatible classifier.
        sampler: Optional imblearn resampler (e.g. ``SMOTE``) inserted between
            preprocessing and the model. ``None`` means no resampling.
    """
    steps: list[tuple[str, object]] = [("preprocess", build_preprocessor())]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("model", model))
    return ImbPipeline(steps)


def describe_features() -> pd.DataFrame:
    """Return a human-readable table describing every model input feature."""
    rows = [
        ("air_temperature_k", "numeric", "Air temperature [K]", "Ambient air temperature around the machine."),
        ("process_temperature_k", "numeric", "Process temperature [K]", "Temperature of the process itself."),
        ("rotational_speed_rpm", "numeric", "Rotational speed [rpm]", "Shaft rotation speed."),
        ("torque_nm", "numeric", "Torque [Nm]", "Torque applied to the shaft."),
        ("tool_wear_min", "numeric", "Tool wear [min]", "Accumulated tool wear time."),
        ("power_kw", "derived", "torque x speed", "Mechanical power proxy."),
        ("temp_diff_k", "derived", "process - air temp.", "Process heating relative to ambient air."),
        ("type", "categorical", "Product type (L/M/H)", "Low, medium or high quality product variant."),
    ]
    return pd.DataFrame(rows, columns=["feature", "kind", "source", "description"])
