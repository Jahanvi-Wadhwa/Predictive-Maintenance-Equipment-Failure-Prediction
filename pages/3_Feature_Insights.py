"""Streamlit page - Feature Insights (explainability + exploratory charts).

Importance values come from ``models/artifacts/permutation_importance.csv``
(computed on the real held-out test set). Exploratory charts are regenerated
from the real dataset when missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import streamlit as st

from src.dashboard_utils import (
    AMBER,
    NAVY,
    SLATE,
    TEAL,
    inject_css,
    new_fig,
    note,
    page_header,
    show_figure,
)
from src.config import FIGURES_DIR, PERM_IMPORTANCE_PATH, RAW_DATA_PATH
from src.data import DatasetError, load_dataset
from src.eda import run_eda

st.set_page_config(
    page_title="Predictive Maintenance | Feature Insights",
    page_icon=" ",
    layout="wide",
)
inject_css()

st.sidebar.markdown("##   Predictive Maintenance")
st.sidebar.markdown(
    "Which operating conditions influence predictions, and how machines in "
    "the dataset differ between failed and healthy states."
)

page_header(
    "Feature Insights",
    "What drives the model's predictions - and what we can observe in the data.",
)

# Plain-English glosses for each model input feature.
FEATURE_NOTES = {
    "rotational_speed_rpm": (
        "**Rotational speed** - failing machines in this dataset often run at "
        "unusually *low* speed under load (the classic stall/overload "
        "signature). Shuffling it costs the model the most average precision, "
        "so it carries the strongest individual signal."
    ),
    "temp_diff_k": (
        "**Process - air temperature gap (derived)** - how much the process "
        "heats relative to ambient air. A shrinking gap means less useful "
        "heat is being converted to work - a sign of inefficiency/fault. "
        "Engineered from two sensors, it often outperforms either raw "
        "temperature alone."
    ),
    "tool_wear_min": (
        "**Tool wear** - accumulated wear time. Wear-related failures (TWF, "
        "OSF in the dataset) become much more likely late in a tool's life."
    ),
    "power_kw": (
        "**Mechanical power proxy (torque x speed, derived)** - unusual power "
        "draw for a given product type indicates abnormal loading."
    ),
    "torque_nm": (
        "**Torque** - higher torque at a given speed raises stress on the "
        "shaft and bearings; combined with speed it drives the power proxy."
    ),
    "type": (
        "**Product type (L/M/H)** - weaker but non-zero: failure risk differs "
        "slightly across product variants in this dataset."
    ),
    "process_temperature_k": (
        "**Process temperature on its own** adds little once the temperature "
        "*gap* is already in the model - its importance hovers around zero."
    ),
    "air_temperature_k": (
        "**Air temperature on its own** likewise contributes almost nothing "
        "beyond what the derived gap already encodes; small negative values "
        "just mean shuffling it does not hurt (noise)."
    ),
}

# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------
st.markdown("###   1. Permutation importance (model-agnostic)")

if not PERM_IMPORTANCE_PATH.exists():
    st.warning(
        "`models/artifacts/permutation_importance.csv` not found - "
        "run `python -m src.train` first."
    )
    st.stop()

perm = pd.read_csv(PERM_IMPORTANCE_PATH)

fig, ax = new_fig(7.6, 5.0)
ordered = perm.iloc[::-1]
colors = [TEAL if v >= 0 else AMBER for v in ordered["importance_mean"]]
ax.barh(ordered["feature"], ordered["importance_mean"], xerr=ordered["importance_std"],
        color=colors, ecolor=SLATE, capsize=3)
ax.axvline(0, color=SLATE, linewidth=1)
ax.set_xlabel("Mean drop in average precision when the feature is shuffled")
ax.set_title("Permutation importance - selected model, held-out test set",
             fontsize=12, fontweight="bold")
show_figure(fig)

top = perm.iloc[0]
note(
    f"**Method:** each feature's values are randomly shuffled in the test set "
    f"10 times; the bar shows how much the average precision drops "
    f"(error bars = standard deviation). The top feature "
    f"**{top['feature']}** costs **{top['importance_mean']:.3f}** average "
    f"precision when scrambled, so the model leans on it heavily. "
    "**Importance tells you what the model relies on - it does *not* prove "
    "that a feature *causes* failures.**"
)

st.markdown("#### Plain-English reading")
for feature in perm["feature"]:
    note_text = FEATURE_NOTES.get(feature, feature)
    st.markdown(f"- {note_text}")

# ---------------------------------------------------------------------------
# Exploratory charts from the real dataset
# ---------------------------------------------------------------------------
st.markdown("###   2. Exploratory charts (real dataset)")

fig_files = {
    "Class distribution": FIGURES_DIR / "eda_class_distribution.png",
    "Feature distributions by failure status": FIGURES_DIR / "eda_feature_distributions.png",
    "Correlation heatmap": FIGURES_DIR / "eda_correlation.png",
    "Failure modes among failed machines": FIGURES_DIR / "eda_failure_modes.png",
}

if not all(path.exists() for path in fig_files.values()):
    try:
        with st.spinner("Generating EDA figures from the dataset ..."):
            run_eda(write_report=False)
    except DatasetError as exc:
        st.error(
            "**Dataset unavailable.** Run `python -m src.data` to download it, "
            f"or place `ai4i2020.csv` in `data/raw/`.\n\n_{exc}_"
        )
        st.stop()

tabs = st.tabs(list(fig_files))
for tab, (label, path) in zip(tabs, fig_files.items()):
    with tab:
        if path.exists():
            st.image(str(path))
        else:
            st.caption("Figure not available - run `python -m src.eda`.")

st.markdown(
    "**What to look for:**\n\n"
    "- **Class distribution** - roughly 30 healthy machines per failure, "
    "which is exactly why accuracy is discarded as a headline metric.\n"
    "- **Distributions** - failed machines skew toward higher torque, higher "
    "tool wear and lower rotational speed; overlaps mean no single threshold "
    "can separate the classes (hence a probabilistic model).\n"
    "- **Correlations** - air/process temperature and the derived features "
    "correlate with each other; none of them is individually close to the "
    "target, so the model must combine weak signals.\n"
    "- **Failure modes** - Tool Wear Failure (TWF) and Overstrain Failure "
    "(OSF) dominate; several machines show more than one mode, which is why "
    "these columns must never be model inputs."
)

note(
    "Reminder: all charts above are generated from "
    "`data/raw/ai4i2020.csv` on this machine - regenerate them any time with "
    "`python -m src.eda`."
)
