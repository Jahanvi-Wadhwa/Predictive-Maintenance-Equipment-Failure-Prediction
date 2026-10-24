"""Central configuration for the predictive maintenance project.

All paths, random seeds, column names and preprocessing decisions live here so
that the training pipeline, the dashboard and the tests share one source of
truth.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Paths (all paths are relative to the repository root, never machine-specific)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "ai4i2020.csv"
MODELS_DIR = ROOT_DIR / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MODEL_PATH = MODELS_DIR / "failure_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
PERM_IMPORTANCE_PATH = ARTIFACTS_DIR / "permutation_importance.csv"
REFERENCE_STATS_PATH = ARTIFACTS_DIR / "reference_stats.json"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository.
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
DATASET_NAME = "ai4i2020.csv"
# SHA-256 of the ai4i2020.csv file inside the UCI archive (verified at download).
DATASET_SHA256 = "dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e"

TARGET_COLUMN = "Machine failure"

# Identifier columns carry no predictive information (row id / product code)
# and are removed before modelling.
ID_COLUMNS = ["UDI", "Product ID"]

# The five failure-mode columns (TWF, HDF, PWF, OSF, RNF) are *components* of
# the target: a machine has failed if any of them is 1. Keeping them as input
# features would be direct target leakage, so they are excluded from modelling
# but kept for exploratory analysis (e.g. failure-mode breakdown).
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Raw dataset feature names -> clean snake_case names used everywhere else.
RAW_TO_CLEAN = {
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Type": "type",
}

NUMERIC_FEATURES = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    # Two derived features (see src.features.add_engineered_features)
    "power_kw",
    "temp_diff_k",
]

CATEGORICAL_FEATURES = ["type"]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
TEST_SIZE = 0.2
CV_FOLDS = 5
CV_SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "recall": "recall",
    "f1": "f1",
}

# Risk categories (Low / Moderate / High) are derived from out-of-fold training
# probabilities (saved in metrics.json):
#   * "moderate" = lowest probability cut-off on out-of-fold training
#     predictions that reaches PRECISION_TARGET while keeping recall at or
#     above RECALL_TARGET (a real operating point, not a round number).
#   * "high"     = 0.5, the model's own default classification threshold.
# They are relative risk bands for this model and dataset only - they are
# NOT validated industrial safety limits.
PRECISION_TARGET = 0.70
RECALL_TARGET = 0.80
HIGH_RISK_THRESHOLD = 0.5
