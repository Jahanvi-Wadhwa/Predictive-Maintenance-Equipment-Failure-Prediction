"""End-to-end dashboard checks using Streamlit's built-in ``AppTest``.

These execute the real dashboard scripts headlessly (no browser needed) and
fail if any page raises, or if the Overview page shows numbers that do not
match ``models/artifacts/metrics.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from streamlit.testing.v1 import AppTest

from src.config import METRICS_PATH, MODEL_PATH

TIMEOUT = 90

DASHBOARD_PAGES = [
    ROOT / "app.py",
    ROOT / "pages" / "1_Failure_Risk_Prediction.py",
    ROOT / "pages" / "2_Model_Evaluation.py",
    ROOT / "pages" / "3_Feature_Insights.py",
    ROOT / "pages" / "4_Sample_Analysis.py",
]

requires_artifacts = pytest.mark.skipif(
    not (METRICS_PATH.exists() and MODEL_PATH.exists()),
    reason="run `python -m src.train` first",
)


def run_app(path: Path) -> AppTest:
    at = AppTest.from_file(str(path), default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, f"{path.name} raised:\n{at.exception}"
    assert not [e.value for e in at.error], f"{path.name} rendered st.error"
    return at


@pytest.mark.parametrize("page", DASHBOARD_PAGES, ids=lambda p: p.name)
def test_page_executes_without_exceptions(page: Path):
    if not page.exists():
        pytest.fail(f"missing dashboard page: {page}")
    run_app(page)


@requires_artifacts
def test_overview_shows_real_saved_metrics():
    at = run_app(ROOT / "app.py")
    payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    selected = payload["selected_model"]
    m = payload["models"][selected]["test"]["metrics"]

    metrics_by_label = {el.label: el.value for el in at.metric}
    assert "ROC-AUC" in metrics_by_label
    assert metrics_by_label["ROC-AUC"] == f"{m['roc_auc']:.3f}"
    assert metrics_by_label["Recall (failures caught)"] == f"{m['recall']:.1%}"
    assert metrics_by_label["Failure rate"] == (
        f"{payload['dataset']['failure_rate'] * 100:.2f}%"
    )


@requires_artifacts
def test_prediction_page_returns_probability_on_click():
    at = run_app(ROOT / "pages" / "1_Failure_Risk_Prediction.py")

    # The form exposes 6 inputs (5 numeric + product type) and a predict button.
    assert len(at.number_input) == 5, "expected 5 numeric sensor inputs"
    assert len(at.selectbox) == 1, "expected product-type selector"
    assert len(at.button) >= 1

    at.button[0].click().run(timeout=TIMEOUT)
    assert not at.exception, f"prediction raised:\n{at.exception}"

    rendered = " ".join(str(m.value) for m in at.markdown)
    captions = " ".join(str(c.value) for c in at.caption)
    assert "estimated probability of failure" in rendered
    assert any(badge in rendered for badge in ("Low risk", "Moderate risk", "High risk"))
    assert "not validated industrial safety limits" in captions + rendered


@requires_artifacts
def test_evaluation_page_reports_selected_model():
    at = run_app(ROOT / "pages" / "2_Model_Evaluation.py")
    payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    selected = payload["selected_model"].replace("_", " ").title()
    rendered = " ".join(str(m.value) for m in at.markdown)
    assert f"Selected model: {selected}" in rendered
    assert len(at.dataframe) >= 2, "expected CV and test metric tables"


def test_sample_analysis_page_requires_upload():
    at = run_app(ROOT / "pages" / "4_Sample_Analysis.py")
    assert len(at.file_uploader) == 1
    rendered = " ".join(str(m.value) for m in at.markdown)
    assert "Upload a CSV" in rendered