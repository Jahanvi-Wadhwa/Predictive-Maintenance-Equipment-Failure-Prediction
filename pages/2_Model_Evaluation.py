"""Streamlit page - Model Evaluation.

Every number and curve on this page is read from
``models/artifacts/metrics.json``, written by ``python -m src.train``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard_utils import (
    AMBER,
    NAVY,
    TEAL,
    confusion_matrix_figure,
    fmt_metric,
    get_metrics,
    inject_css,
    missing_artifacts_warning,
    new_fig,
    note,
    page_header,
    show_figure,
)
from src.dashboard_utils import show_dataframe
from src.evaluate import trapz

st.set_page_config(
    page_title="Predictive Maintenance | Model Evaluation",
    page_icon=" ",
    layout="wide",
)
inject_css()

st.sidebar.markdown("##   Predictive Maintenance")
st.sidebar.markdown(
    "Model comparison, ROC / precision-recall curves and confusion matrices "
    "- all computed on real held-out data."
)

page_header(
    "Model Evaluation",
    "Three classifiers compared with cross-validation, then scored once on "
    "an untouched test set.",
)

metrics = get_metrics()
if missing_artifacts_warning(metrics):
    st.stop()

selected = metrics["selected_model"]
models = metrics["models"]
DISPLAY = {name: name.replace("_", " ").title() for name in models}
COLORS = {"Logistic Regression": NAVY, "Random Forest": TEAL,
          "Hist Gradient Boosting": AMBER}

# ---------------------------------------------------------------------------
# Cross-validation comparison
# ---------------------------------------------------------------------------
st.markdown("###   1. Cross-validation (training set, 5-fold stratified)")
st.caption(
    f"{metrics['cv_folds']}-fold stratified CV on the {metrics['split']['n_train']:,} "
    "training rows only. Preprocessing and SMOTE are refit inside every fold, "
    "so validation rows never influence their own fold's model."
)

cv_rows = []
for name, res in models.items():
    cv = res["cv"]
    cv_rows.append({
        "Model": DISPLAY[name],
        "ROC-AUC": f"{cv['roc_auc_mean']:.4f} ± {cv['roc_auc_std']:.3f}",
        "Avg. precision (PR-AUC)": f"{cv['average_precision_mean']:.4f} ± {cv['average_precision_std']:.3f}",
        "Recall": f"{cv['recall_mean']:.4f} ± {cv['recall_std']:.3f}",
        "F1": f"{cv['f1_mean']:.4f} ± {cv['f1_std']:.3f}",
        "SMOTE": "yes" if res["uses_smote"] else "no (class weights)",
    })
show_dataframe(pd.DataFrame(cv_rows), hide_index=True)

fig, ax = new_fig(7.5, 4.2)
names = list(DISPLAY.values())
x = np.arange(len(names))
width = 0.36
roc = [models[n]["cv"]["roc_auc_mean"] for n in models]
ap = [models[n]["cv"]["average_precision_mean"] for n in models]
b1 = ax.bar(x - width / 2, roc, width, label="ROC-AUC", color=NAVY)
b2 = ax.bar(x + width / 2, ap, width, label="Avg. precision", color=TEAL)
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=8, color=NAVY)
ax.set_xticks(x, names, fontsize=9)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Mean CV score")
ax.set_title("Cross-validation model comparison", fontsize=12, fontweight="bold")
ax.legend(frameon=False, fontsize=9)
show_figure(fig)

# ---------------------------------------------------------------------------
# Test-set metrics
# ---------------------------------------------------------------------------
st.markdown("###   2. Held-out test set (never used for selection)")
test_rows = []
for name, res in models.items():
    m = res["test"]["metrics"]
    test_rows.append({
        "Model": DISPLAY[name],
        "Accuracy": fmt_metric(m["accuracy"]),
        "Precision": fmt_metric(m["precision"]),
        "Recall": fmt_metric(m["recall"]),
        "F1": fmt_metric(m["f1"]),
        "ROC-AUC": fmt_metric(m["roc_auc"]),
        "Avg. precision": fmt_metric(m["average_precision"]),
        "Flagged correct (TP)": int(np.asarray(m["confusion_matrix"])[1, 1]),
        "Missed failures (FN)": int(np.asarray(m["confusion_matrix"])[1, 0]),
        "False alarms (FP)": int(np.asarray(m["confusion_matrix"])[0, 1]),
    })
test_df = pd.DataFrame(test_rows)
show_dataframe(test_df, hide_index=True)

sel_metrics = models[selected]["test"]["metrics"]
best = test_df.loc[test_df["Model"] == DISPLAY[selected]].iloc[0]
st.markdown(
    f"**Selected model: {DISPLAY[selected]}** - {metrics['selection_criterion']}"
)
st.markdown(
    f"Its test-set ROC-AUC is **{best['ROC-AUC']}**, average precision "
    f"**{best['Avg. precision']}**, recall **{best['Recall']}** and F1 "
    f"**{best['F1']}**, catching **{best['Flagged correct (TP)']} of "
    f"{int(np.asarray(sel_metrics['confusion_matrix'])[1].sum())} real "
    "failures with "
    f"**{best['False alarms (FP)']} false alarms**. Random Forest scored a "
    "similar ROC-AUC with fewer false alarms but missed more true failures; "
    "Logistic Regression's linear boundary misses subtle non-linear "
    "interactions, so its precision collapses on this imbalanced target."
)

# ---------------------------------------------------------------------------
# ROC and precision-recall curves
# ---------------------------------------------------------------------------
st.markdown("###   3. ROC and precision-recall curves")
roc_col, pr_col = st.columns(2)

with roc_col:
    fig, ax = new_fig(6.0, 4.8)
    for name in models:
        c = models[name]["test"]["curves"]
        label = DISPLAY[name]
        auc = trapz(c["tpr"], c["fpr"])
        ax.plot(c["fpr"], c["tpr"], color=COLORS.get(label, NAVY), linewidth=1.8,
                label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("Recall (true positive rate)")
    ax.set_title("ROC curves - test set", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    show_figure(fig)

with pr_col:
    fig, ax = new_fig(6.0, 4.8)
    baseline = metrics["dataset"]["failure_rate"]
    for name in models:
        c = models[name]["test"]["curves"]
        label = DISPLAY[name]
        ap = trapz(c["precision"], c["recall"])
        ax.plot(c["recall"], c["precision"], color=COLORS.get(label, NAVY),
                linewidth=1.8, label=f"{label} (AP = {ap:.3f})")
    ax.axhline(baseline, linestyle="--", color="#94a3b8", linewidth=1,
               label=f"Baseline = failure rate ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves - test set", fontsize=12,
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    show_figure(fig)

note(
    "**Why both curves?** ROC-AUC can look optimistic when negatives "
    "outnumber positives 30:1. The precision-recall curve is the stricter, "
    "more informative view for a rare event like machine failure - the flat "
    "teal line marks the \"random guess\" baseline of "
    f"{baseline:.3f}."
)

# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------
st.markdown("###   4. Confusion matrices")
cols = st.columns(len(models))
for col, (name, res) in zip(cols, models.items()):
    with col:
        st.caption(f"**{DISPLAY[name]}**")
        show_figure(
            confusion_matrix_figure(
                res["test"]["metrics"]["confusion_matrix"],
                f"Test set ({metrics['split']['n_test']:,} rows)",
            )
        )

m = sel_metrics
tn, fp, fn, tp = np.asarray(m["confusion_matrix"]).ravel()
st.markdown(
    f"**Reading the winning matrix ({DISPLAY[selected]}):** "
    f"**{tp}** machines that really failed were caught; **{fn}** failures "
    f"were missed (a missed failure risks unplanned downtime - the cost the "
    f"project prioritises avoiding); **{fp}** healthy machines were flagged, "
    f"costing an unnecessary but harmless inspection; **{tn}** healthy "
    f"machines were correctly left alone."
)

# ---------------------------------------------------------------------------
# Selection rationale
# ---------------------------------------------------------------------------
with st.expander("Which model was selected, and why?"):
    st.markdown(
        f"- **Criterion:** {metrics['selection_criterion']}\n"
        f"- **Winner:** {DISPLAY[selected]} with CV average precision "
        f"**{models[selected]['cv']['average_precision_mean']:.4f}**.\n"
        "- Average precision (area under the precision-recall curve) is used "
        "instead of accuracy because only ~3.4% of rows are failures; it "
        "rewards catching failures without drowning in false alarms.\n"
        "- The test set was evaluated once per model *after* selection and "
        "plays no part in choosing the winner or tuning thresholds.\n"
        "- Hyperparameters are fixed, published defaults rather than "
        "exhaustively tuned - a deliberate choice to keep the project "
        "reproducible and avoid test-set overfitting."
    )