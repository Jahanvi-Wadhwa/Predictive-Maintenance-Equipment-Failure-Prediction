"""Streamlit page - Sample Analysis (batch CSV scoring).

Users upload a CSV with the sensor columns (raw AI4I headers or snake_case
names both work); every row is validated and scored with the same
:func:`src.predict.predict_batch` used everywhere else, and the result is
offered back as a downloadable CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.dashboard_utils import (
    AMBER,
    NAVY,
    TEAL,
    get_bundle,
    inject_css,
    note,
    page_header,
    show_dataframe,
)
from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN
from src.predict import InputValidationError, normalize_columns, predict_batch

st.set_page_config(
    page_title="Predictive Maintenance | Sample Analysis",
    page_icon=" ",
    layout="wide",
)
inject_css()

st.sidebar.markdown("##   Predictive Maintenance")
st.sidebar.markdown(
    "Upload a CSV of machine readings, score every row with the trained "
    "model, and download the results."
)

page_header(
    "Sample Analysis",
    "Batch-predict failure risk for many machines at once from a CSV file.",
)

bundle = get_bundle()
if bundle is None:
    st.stop()

# ---------------------------------------------------------------------------
# Expected schema
# ---------------------------------------------------------------------------
with st.expander("Expected CSV format (click to expand)"):
    st.markdown(
        "Required columns (raw AI4I headers **or** snake_case names):\n\n"
        "| Column | Type | Example |\n|---|---|---|\n"
        "| `air_temperature_k` (`Air temperature [K]`) | number | 298.1 |\n"
        "| `process_temperature_k` (`Process temperature [K]`) | number | 308.6 |\n"
        "| `rotational_speed_rpm` (`Rotational speed [rpm]`) | number | 1551 |\n"
        "| `torque_nm` (`Torque [Nm]`) | number | 42.8 |\n"
        "| `tool_wear_min` (`Tool wear [min]`) | number | 0 |\n"
        "| `type` (`Type`) | L / M / H | M |\n\n"
        "Extra columns are ignored. Missing or non-numeric values are "
        "reported instead of guessed. Two derived features "
        "(`power_kw`, `temp_diff_k`) are computed automatically.\n\n"
        "The fastest way to produce a valid file: open "
        "`data/raw/ai4i2020.csv` and upload it as-is."
    )

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.info(
        "Upload a CSV to generate batch predictions. A ready-made sample "
        "lives at `data/raw/ai4i2020.csv`."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Load + validate
# ---------------------------------------------------------------------------
try:
    raw_df = pd.read_csv(uploaded)
except Exception as exc:  # noqa: BLE001
    st.error(f"**Could not parse the file as CSV:** {exc}")
    st.stop()

raw_df.columns = [str(c).strip() for c in raw_df.columns]
df = normalize_columns(raw_df)

if df.empty:
    st.error("The uploaded file has no data rows.")
    st.stop()

required = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
derived = {"power_kw", "temp_diff_k"}
required_user = [c for c in required if c not in derived]
missing = [c for c in required_user if c not in df.columns]

st.caption(
    f"Loaded **{len(df):,}** rows and **{df.shape[1]}** columns. "
    f"Detected model inputs: "
    f"{', '.join(c for c in required_user if c in df.columns) or 'none'}."
)

if missing:
    st.error(
        f"**Missing required column(s):** {', '.join(f'`{m}`' for m in missing)}\n\n"
        "Accepted aliases are shown in the *Expected CSV format* box above - "
        "for example `Air temperature [K]` also works."
    )
    st.stop()

# Quick dtype preview before scoring
st.markdown("###   Input preview")
show_dataframe(df.head(10), hide_index=True)

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if st.button("Generate predictions", type="primary"):
    try:
        results = predict_batch(df, bundle)
    except InputValidationError as exc:
        st.error(f"**Validation failed:** {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001 - model-loading/runtime guard
        st.error(f"**Prediction failed:** {type(exc).__name__}: {exc}")
        st.stop()

    n_flagged = int(results["prediction"].sum())
    total = len(results)

    st.markdown("###   Results")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Rows scored", f"{total:,}")
    s2.metric("Flagged as failure", f"{n_flagged:,}")
    s3.metric("High risk", int((results["risk_category"] == "High").sum()))
    s4.metric(
        "Mean failure probability", f"{results['failure_probability'].mean():.4f}"
    )

    if n_flagged:
        st.markdown(
            f"**{n_flagged} of {total:,} machines** "
            f"({n_flagged / total:.1%}) crossed the model's 0.50 decision "
            "threshold and should be reviewed first. Ranked most-risky first:"
        )
        flagged = (
            results[results["prediction"] == 1]
            .sort_values("failure_probability", ascending=False)
        )
        show_dataframe(flagged, hide_index=True)
    else:
        st.success(
            "No rows crossed the 0.50 decision threshold. The highest "
            f"probability observed was "
            f"{results['failure_probability'].max():.4f}."
        )

    # Risk distribution chart
    counts = results["risk_category"].value_counts().reindex(
        ["Low", "Moderate", "High"]
    ).fillna(0)
    import matplotlib

    matplotlib.use("Agg")
    from src.dashboard_utils import new_fig, show_figure

    fig, ax = new_fig(6.4, 3.4)
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[TEAL, AMBER, NAVY],
        width=0.5,
        edgecolor="white",
    )
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{int(value):,}",
            ha="center",
            va="bottom",
            color=NAVY,
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylabel("Rows")
    ax.set_title("Risk category distribution", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(counts.max(), 1) * 1.2)
    show_figure(fig)

    st.markdown("###   Download")
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions CSV",
        data=csv_bytes,
        file_name="failure_predictions.csv",
        mime="text/csv",
        type="primary",
    )

    note(
        "Columns added: `failure_probability` (model score), `prediction` "
        "(1 = flagged at the 0.50 cut-off) and `risk_category` "
        "(Low / Moderate / High relative risk band). "
        f"Target column `{TARGET_COLUMN}`, if present in your file, is never "
        "used as an input - predictions are computed from the sensor columns "
        "only."
    )
