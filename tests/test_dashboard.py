"""Dashboard checks that do not require a running Streamlit server.

Full interactive verification is done by launching the app (see README);
here we guarantee every dashboard file at least parses and that the shared
helpers import cleanly.
"""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DASHBOARD_FILES = [
    ROOT / "app.py",
    *sorted((ROOT / "pages").glob("*.py")),
]


def test_dashboard_has_expected_pages():
    names = {f.name for f in DASHBOARD_FILES}
    assert "app.py" in names
    assert len(list((ROOT / "pages").glob("*.py"))) == 4, "expected 4 dashboard pages"


def test_every_dashboard_file_compiles():
    for path in DASHBOARD_FILES:
        assert path.exists(), f"missing dashboard file {path}"
        # py_compile raises on syntax errors.
        py_compile.compile(str(path), doraise=True, cfile=str(Path("/tmp") / (path.stem + ".pyc")))


def test_dashboard_utils_imports_and_palette():
    from src.dashboard_utils import (
        NAVY,
        RISK_COLORS,
        SLATE,
        TEAL,
        fmt_metric,
        fmt_prob,
        risk_badge,
    )

    assert NAVY.startswith("#") and TEAL.startswith("#") and SLATE.startswith("#")
    assert set(RISK_COLORS) == {"Low", "Moderate", "High"}
    assert fmt_prob(0.00001) == "< 0.0001"
    assert fmt_prob(0.1234) == "0.123"
    assert fmt_metric(None) == "n/a"
    assert fmt_metric(0.98765, digits=2) == "0.99"


def test_pages_do_not_import_training_logic():
    """Prediction pages must use src.predict - never re-fit or hardcode models."""
    forbidden = ["cross_val_predict", "GridSearch", ".fit(", "RandomForestClassifier("]
    for path in (ROOT / "pages").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{path.name} contains training token: {token}"


def test_no_hardcoded_fake_metrics_in_pages():
    """Eval numbers must come from metrics.json, not literals."""
    suspicious = ["roc_auc = 0.", "f1 = 0.", "accuracy = 0.9"]
    for path in (ROOT / "pages").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in suspicious:
            assert token not in source, f"{path.name} hardcodes {token}"