"""Shared pytest fixtures.

Tests run against the *real* AI4I dataset (a small slice of it for speed) -
no fabricated data is used to produce any reported result.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from src.config import MODEL_PATH, RAW_DATA_PATH
from src.data import load_dataset


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """The full raw dataset (session-scoped; the CSV ships with the repo)."""
    if not RAW_DATA_PATH.exists():
        pytest.skip(f"Dataset not downloaded: {RAW_DATA_PATH} (run `python -m src.data`)")
    return load_dataset()


@pytest.fixture(scope="session")
def small_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    """A stratified 600-row slice keeping both classes (speeds up model tests)."""
    failed = raw_df[raw_df["Machine failure"] == 1]
    healthy = raw_df[raw_df["Machine failure"] == 0]
    sample = pd.concat([failed.head(60), healthy.head(540)]).sample(
        frac=1, random_state=42
    )
    return sample.reset_index(drop=True)


@pytest.fixture(scope="session")
def trained_bundle():
    """The saved model bundle, or skip if training has not run yet."""
    if not MODEL_PATH.exists():
        pytest.skip("Trained model missing - run `python -m src.train` first")
    from src.predict import load_bundle

    return load_bundle()