"""Streamlit entry point - Overview page.

Launch from the repository root:

    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import streamlit as st

from src.dashboard_utils import (
    AMBER,
    NAVY,
    SLATE,
    TEAL,
    card,
    fmt_metric,
    get_metrics,
    inject_css,
    missing_artifacts_warning,
    new_fig,
    note,
    page_header,
    show_figure,
)

st.set_page_config(
    page_title="Predictive Maintenance | Overview",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ---------------------------------------------------------------------------
# Sidebar navigation description
# ---------------------------------------------------------------------------
st.sidebar.markdown("##   Predictive Maintenance")
st.sidebar.markdown(
    "ML-driven equipment failure risk analysis on the "
    "**AI4I 2020** dataset.\n\n"
    "- **Overview** - project, data, results\n"
    "- **Failure Risk Prediction** - score a single machine\n"
    "- **Model Evaluation** - metrics & confusion matrices\n"
    "- **Feature Insights** - what drives predictions\n"
    "- **Sample Analysis** - batch score a CSV"
)
st.sidebar.markdown(
    "---\n"
    " *Student portfolio project. "
    "Predictions are offline estimates from sensor readings, "
    "not a certified monitoring system.*"
)

page_header(
    "Predictive Maintenance - Equipment Failure Prediction",
    "An end-to-end machine learning workflow that estimates the risk of "
    "industrial machine failure from operating conditions.",
)

metrics = get_metrics()
if missing_artifacts_warning(metrics):
    st.stop()

# ---------------------------------------------------------------------------
# A. Project purpose
# ---------------------------------------------------------------------------
st.markdown("###   Project purpose")
st.markdown(
    "Unexpected equipment failure stops production lines and can be "
    "dangerous. **Predictive maintenance** uses current operating conditions to "
    "estimate how likely a machine is to fail soon, so maintenance teams can "
    "act *before* a breakdown instead of after one.\n\n"
    "This project walks through a complete, reproducible ML workflow: "
    "data exploration, leakage-safe preprocessing, class-imbalance handling, "
    "cross-validated model comparison, evaluation on an untouched test set, "
    "model-agnostic explainability, and an interactive dashboard."
)

# ---------------------------------------------------------------------------
# B. Dataset summary
# ---------------------------------------------------------------------------
st.markdown("###   Dataset")
ds = metrics["dataset"]
split = metrics["split"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{ds['n_rows']:,}")
c2.metric("Model features", f"{len(ds['features'])}")
c3.metric("Failures", f"{ds['n_failures']:,}")
c4.metric("Failure rate", f"{ds['failure_rate'] * 100:.2f}%")

st.markdown(
    f"**Source:** [{ds['name']}]({ds['url']}) - {ds['source']}. "
    f"The dataset simulates 10,000 industrial machine records with five "
    f"sensors (air temperature, process temperature, rotational speed, torque, "
    f"tool wear) and a product type (L/M/H). A stratified split held out "
    f"**{split['n_test']:,} rows ({split['test_failures']} real failures)** "
    f"as an untouched test set; everything - preprocessing, SMOTE, model "
    f"selection - was fit on the remaining {split['n_train']:,} rows only."
)

# ---------------------------------------------------------------------------
# C. Failure-class distribution (real counts from the dataset)
# ---------------------------------------------------------------------------
st.markdown("###   Failure-class distribution")
col_chart, col_text = st.columns([1.05, 1])

with col_chart:
    fig, ax = new_fig(6.2, 3.6)
    counts = [ds["n_rows"] - ds["n_failures"], ds["n_failures"]]
    bars = ax.bar(["No failure", "Failure"], counts,
                  color=[TEAL, AMBER], width=0.5, edgecolor="white")
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}",
                ha="center", va="bottom", color=NAVY, fontsize=11, fontweight="bold")
    ax.set_ylabel("Machines")
    ax.set_title("Class imbalance: only ~3.4% of records are failures",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.18)
    show_figure(fig)

with col_text:
    st.markdown(
        f"With a **{ds['failure_rate'] * 100:.1f}%** positive rate, a model "
        "that always predicted *\"no failure\"* would be "
        f"**{100 - ds['failure_rate'] * 100:.1f}% accurate** while detecting "
        "**zero** failures - accuracy is misleading here.\n\n"
        "The pipeline therefore handles imbalance explicitly "
        "(SMOTE and/or balanced class weights, applied only inside training "
        "folds) and is judged on **precision, recall, F1, ROC-AUC and "
        "average precision (PR-AUC)**."
    )
    note(
        "**Target leakage control:** the columns `TWF, HDF, PWF, OSF, RNF` are "
        "the *components* of the target (`Machine failure = any of them`) and "
        "`UDI`/`Product ID` are identifiers - all are excluded from the model "
        "inputs."
    )

# ---------------------------------------------------------------------------
# D. Key evaluation metrics (actual test-set results)
# ---------------------------------------------------------------------------
st.markdown("###   Key model evaluation (held-out test set)")
selected = metrics["selected_model"]
sel_res = metrics["models"][selected]
m = sel_res["test"]["metrics"]
tn, fp, fn, tp = np.asarray(m["confusion_matrix"]).ravel()

st.caption(
    f"Selected model: **{selected.replace('_', ' ').title()}** - "
    f"{metrics['selection_criterion']}"
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("ROC-AUC", fmt_metric(m["roc_auc"]))
k2.metric("PR-AUC (avg. precision)", fmt_metric(m["average_precision"]))
k3.metric("Recall (failures caught)", f"{m['recall']:.1%}")
k4.metric("Precision", f"{m['precision']:.1%}")
k5.metric("F1-score", fmt_metric(m["f1"]))

e1, e2, e3 = st.columns(3)
e1.metric("Failures detected (TP)", f"{tp} / {tp + fn}")
e2.metric("Missed failures (FN)", f"{fn}")
e3.metric("False alarms (FP)", f"{fp}")

st.markdown(
    f"Out of **{tp + fn} real failures** in the test set, the model caught "
    f"**{tp}** and missed **{fn}**; it raised **{fp}** false alarms among "
    f"**{tn + fp}** healthy machines. "
    "The trade-off matters in maintenance terms:"
)
card(
    "False negative (missed failure)",
    "A machine that will fail but is rated low risk. The consequence is an "
    "unexpected breakdown, unplanned downtime and possible secondary damage - "
    "the costliest error type, which is why recall is prioritised.",
)
card(
    "False positive (false alarm)",
    "A healthy machine flagged as risky. The consequence is an unnecessary "
    "inspection - an accepted cost of a conservative maintenance policy.",
)

# ---------------------------------------------------------------------------
# E. What it can and cannot do
# ---------------------------------------------------------------------------
st.markdown("###   Scope: what this system can and cannot predict")
col_can, col_cannot = st.columns(2)
with col_can:
    st.markdown(
        "**It CAN**\n\n"
        "- Estimate the probability that a machine *resembling the AI4I data* "
        "is in a pre-failure condition, given five operating readings\n"
        "- Rank risky machines among a batch (Sample Analysis page)\n"
        "- Show which measurements drive each prediction "
        "(Feature Insights page)\n"
        "- Reproduce every reported number via `python -m src.train`"
    )
with col_cannot:
    st.markdown(
        "**It CANNOT**\n\n"
        "- Predict *when* a failure will happen (no time-to-failure modelling)\n"
        "- Detect failures outside the AI4I data-generating process "
        "(it is simulated data, not a live plant)\n"
        "- Guarantee calibrated probabilities or act as a safety system "
        "- risk bands are project thresholds, **not** validated industrial "
        "safety limits\n"
        "- Connect to real sensors - inputs are typed or uploaded by the user"
    )
note(
    "Limitations and future improvements are listed in the README. "
    "Feature importance shown in this dashboard reflects model behaviour, "
    "not causation."
)
