"""Streamlit page - Failure Risk Prediction (single machine)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.dashboard_utils import (
    fmt_prob,
    get_bundle,
    get_metrics,
    get_reference_stats,
    inject_css,
    note,
    page_header,
    risk_badge,
)
from src.predict import InputValidationError, predict_one

st.set_page_config(
    page_title="Predictive Maintenance | Failure Risk Prediction",
    page_icon=" ",
    layout="wide",
)
inject_css()

st.sidebar.markdown("##   Predictive Maintenance")
st.sidebar.markdown(
    "Enter current operating readings to estimate failure risk.\n\n"
    "Defaults and valid ranges come from the real AI4I dataset."
)

page_header(
    "Failure Risk Prediction",
    "Score a single machine from its current operating conditions.",
)

bundle = get_bundle()
if bundle is None:
    st.stop()

metrics = get_metrics()
stats = get_reference_stats() or {}
thresholds = bundle.get("risk_thresholds", {})
model_title = bundle.get("model_name", "model").replace("_", " ").title()


def number_field(feature: str, label: str, unit: str, step: float = 0.1) -> float:
    """Number input defaulted to the dataset median, bounded by observed data."""
    ref = stats.get(feature, {})
    vmin = float(ref.get("min", 0.0))
    vmax = float(ref.get("max", 100.0))
    default = float(ref.get("median", (vmin + vmax) / 2))
    span = vmax - vmin
    return st.number_input(
        f"{label} ({unit})",
        min_value=round(vmin - 0.05 * span, 3),
        max_value=round(vmax + 0.05 * span, 3),
        value=round(default, 3),
        step=step,
        help=f"Observed in the dataset: {vmin:.1f} - {vmax:.1f} {unit}. "
        f"Default = dataset median ({default:.1f}).",
    )


# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.markdown("###   Machine readings")
left, right = st.columns(2)

with left:
    air_t = number_field("air_temperature_k", "Air temperature", "K", step=0.1)
    proc_t = number_field("process_temperature_k", "Process temperature", "K", step=0.1)
    speed = number_field(
        "rotational_speed_rpm", "Rotational speed", "rpm", step=1.0
    )
with right:
    torque = number_field("torque_nm", "Torque", "Nm", step=0.1)
    wear = number_field("tool_wear_min", "Tool wear", "min", step=1.0)
    product_type = st.selectbox(
        "Product type",
        options=["L", "M", "H"],
        index=1,
        help="L = low, M = medium, H = high quality product variant (from the dataset).",
    )

st.markdown("")
predict_clicked = st.button("Predict failure risk", type="primary")

# ---------------------------------------------------------------------------
# Prediction output
# ---------------------------------------------------------------------------
if predict_clicked:
    reading = {
        "air_temperature_k": air_t,
        "process_temperature_k": proc_t,
        "rotational_speed_rpm": speed,
        "torque_nm": torque,
        "tool_wear_min": wear,
        "type": product_type,
    }
    try:
        result = predict_one(reading, bundle)
    except InputValidationError as exc:
        st.error(f"**Invalid input:** {exc}")
        st.stop()

    probability = result["probability"]
    category = result["risk_category"]
    moderate = float(thresholds.get("moderate", 0.10))
    high = float(thresholds.get("high", 0.50))

    st.markdown("###   Result")
    out_left, out_mid, out_right = st.columns([1.1, 1, 1.4])

    with out_left:
        st.markdown(
            f'<div class="fx-value">{fmt_prob(probability)}</div>'
            "<p style='margin-top:-0.4rem;color:#475569'>"
            "estimated probability of failure</p>",
            unsafe_allow_html=True,
        )
    with out_mid:
        risk_badge(category)
        st.markdown(
            f"<p style='color:#475569;margin-top:0.6rem'>"
            f"Model: <b>{model_title}</b></p>",
            unsafe_allow_html=True,
        )
    with out_right:
        st.markdown(
            f"<p style='color:#475569'>"
            f"<b>Flagged as failure:</b> {'YES' if result['prediction'] else 'no'} "
            f"(model's default 0.50 cut-off)<br>"
            f"<b>Risk bands:</b> Low < {moderate:.2f} &le; Moderate "
            f"< {high:.2f} &le; High</p>",
            unsafe_allow_html=True,
        )

    st.progress(min(max(probability, 0.0), 1.0), text="Failure risk probability")

    # Realized precision/recall of each band on out-of-fold training predictions
    band_stats = (metrics or {}).get("risk_threshold_stats", {})
    high_stats = band_stats.get("high", {})
    mod_stats = band_stats.get("moderate", {})

    # Plain-English interpretation
    if category == "High":
        high_prec = high_stats.get("oof_precision")
        high_txt = (
            f"about {high_prec:.0%}" if high_prec is not None else "typically high"
        )
        interpretation = (
            "The model sees a **strong combination of pre-failure conditions** "
            "(for example high torque with low speed, elevated temperatures or "
            "heavy tool wear). On out-of-fold training predictions, readings at "
            f"or above {high:.2f} were flagged with {high_txt} precision. "
            "Treat this as a **strong signal to inspect the machine soon**, "
            "not a guaranteed failure."
        )
    elif category == "Moderate":
        mod_prec = mod_stats.get("oof_precision")
        mod_txt = (
            f"roughly {mod_prec:.0%}" if mod_prec is not None else "typically"
        )
        interpretation = (
            "The model sees **elevated risk** relative to typical machines in "
            "the training data. Around this cut-off "
            f"({moderate:.2f}) {mod_txt} of flagged machines in training "
            "cross-validation really had failed. A **planned inspection at the "
            "next convenient window** is a reasonable response."
        )
    else:
        interpretation = (
            "The readings resemble the large majority of **healthy machines** "
            f"in the dataset (scores below {moderate:.2f} accounted for most "
            "training records and few failures). This does **not** guarantee "
            "no failure - it means the model found little supporting evidence."
        )
    st.markdown(f"**What this means:** {interpretation}")

    if metrics is not None and band_stats:
        st.caption(
            "Risk bands are project-defined operating points derived from "
            "out-of-fold training predictions; they are "
            "**not validated industrial safety limits**."
        )

    note(
        "**Limitations:** the model was trained on the simulated AI4I 2020 "
        "dataset (10,000 records, 3.4% failure rate), so its probability is "
        "only as good as that data. It does not estimate *when* a failure "
        "would occur, does not adapt to your plant, and should not be used as "
        "a safety instrument. Sensor drift, unmeasured factors (vibration, "
        "lubrication, operator behaviour) and unseen failure modes are outside "
        "its scope."
    )

    with st.expander("How the thresholds were chosen"):
        st.markdown(
            f"- **Moderate ({moderate:.2f})** - the lowest probability cut-off "
            "found on 5-fold out-of-fold training predictions that reaches "
            "70% precision while keeping recall at or above 80%.\n"
            f"- **High ({high:.2f})** - the model's own default classification "
            "threshold (predicted failure probability >= 50%).\n"
            "- Both come from the training set only; the test set was never "
            "used to tune them."
        )
else:
    st.info(
        "Set the operating readings above (defaults are dataset medians) and "
        "click **Predict failure risk**."
    )