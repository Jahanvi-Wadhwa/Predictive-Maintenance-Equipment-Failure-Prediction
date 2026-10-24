"""Dataset download and loading.

The AI4I 2020 Predictive Maintenance Dataset is public and free to download
from the UCI Machine Learning Repository. This module provides:

* :func:`download_dataset` - fetch the CSV programmatically (with a SHA-256
  integrity check) into ``data/raw/``.
* :func:`load_dataset`      - load and validate the CSV for downstream code.

Run the download from the repository root with::

    python -m src.data
"""

from __future__ import annotations

import hashlib
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.config import (
    DATASET_NAME,
    DATASET_SHA256,
    DATASET_URL,
    FAILURE_MODE_COLUMNS,
    ID_COLUMNS,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)

REQUIRED_RAW_COLUMNS = (
    ID_COLUMNS
    + ["Type"]
    + [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    + [TARGET_COLUMN]
    + FAILURE_MODE_COLUMNS
)


class DatasetError(RuntimeError):
    """Raised when the dataset is missing, corrupt or has an unexpected schema."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(dest_path: Path | None = None, force: bool = False) -> Path:
    """Download the AI4I 2020 dataset from UCI into ``data/raw/``.

    The UCI archive is a ZIP that contains ``ai4i2020.csv``. The extracted CSV
    is verified against a known SHA-256 checksum so that a corrupted or
    silently replaced file is detected.

    Args:
        dest_path: Where to write the CSV. Defaults to
            ``data/raw/ai4i2020.csv``.
        force: Re-download even if the file already exists.

    Returns:
        The path to the downloaded CSV.

    Raises:
        DatasetError: If the download fails or the checksum does not match.
    """
    dest_path = Path(dest_path) if dest_path else RAW_DATA_PATH

    if dest_path.exists() and not force:
        checksum = _sha256(dest_path)
        if checksum == DATASET_SHA256:
            print(f"Dataset already present and verified: {dest_path}")
            return dest_path
        print(f"Existing file has unexpected checksum ({checksum[:12]}...), re-downloading.")

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {DATASET_URL} ...")
    try:
        with urllib.request.urlopen(DATASET_URL, timeout=60) as response:
            archive_bytes = response.read()
    except Exception as exc:  # noqa: BLE001 - surface a friendly message
        raise DatasetError(
            f"Could not download the dataset from {DATASET_URL}.\n"
            f"Manual fallback:\n"
            f"  1. Open https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset\n"
            f"  2. Download the dataset zip and extract '{DATASET_NAME}'\n"
            f"  3. Copy it to {dest_path}\n"
            f"Original error: {exc}"
        ) from exc

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise DatasetError("The downloaded archive does not contain a CSV file.")
        # Prefer the canonical file name if several CSVs are present.
        member = next((m for m in members if Path(m).name == DATASET_NAME), members[0])
        with archive.open(member) as src, dest_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    checksum = _sha256(dest_path)
    if checksum != DATASET_SHA256:
        raise DatasetError(
            f"Checksum mismatch after download.\n  expected: {DATASET_SHA256}\n"
            f"  got:      {checksum}\n"
            f"Please download the file manually and place it at {dest_path}"
        )

    print(f"Saved and verified: {dest_path} (sha256 {checksum[:12]}...)")
    return dest_path


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the raw AI4I CSV and validate its schema.

    Args:
        path: CSV path. Defaults to ``data/raw/ai4i2020.csv``.

    Returns:
        The raw dataset as a :class:`pandas.DataFrame`.

    Raises:
        DatasetError: If the file is missing or columns are absent.
    """
    path = Path(path) if path else RAW_DATA_PATH

    if not path.exists():
        raise DatasetError(
            f"Dataset not found at {path}.\n"
            f"Run `python -m src.data` to download it automatically, or place the "
            f"CSV manually at that path (see data/README.md)."
        )

    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(
            f"The dataset at {path} is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    return df


if __name__ == "__main__":
    try:
        download_dataset()
    except DatasetError as err:
        print(f"ERROR: {err}", file=sys.stderr)
        sys.exit(1)
