"""Shared helpers for the Streamlit dashboard.

Keeps a single professional visual style (navy / slate / white / teal) and a
single, error-tolerant loading path for the trained artifacts, so individual
pages stay short and never duplicate training logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.predict import (
    ModelNotLoadedError,
    load_bundle,
    load_metrics,
    load_reference_stats,
)

# Restrained industrial palette
NAVY = "#0f2a4a"
SLATE = "#475569"
TEAL = "#0d9488"
AMBER = "#d97706"
RED = "#b91c1c"
WHITE = "#ffffff"
LIGHT = "#f1f5f9"
BORDER = "#e2e8f0"

RISK_COLORS = {
    "Low": (TEAL, WHITE),
    "Moderate": (AMBER, WHITE),
    "High": (RED, WHITE),
}

CSS = f"""
<style>
    .blockcontainer {{
        max-width: 1140px;
    }}
    header[data-testid="stHeader"] {{
        background: {NAVY};
    }}
    [data-testid="stSidebar"] {{
        background: {NAVY};
    }}
    [data-testid="stSidebar"] .css-1d391kg, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {{
        color: #e2e8f0 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: #cbd5e1;
    }}
    h1, h2, h3 {{
        color: {NAVY} !important;
        font-family: "Source Sans Pro", sans-serif;
        letter-spacing: -0.01em;
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] .css-1d391kg {{
        color: {WHITE} !important;
    }}
    p, li, label {{
        color: {SLATE};
    }}
    .fx-header {{
        border-bottom: 3px solid {TEAL};
        padding-bottom: 0.4rem;
        margin-bottom: 1.2rem;
    }}
    .fx-sub {{
        color: {SLATE};
        font-size: 0.95rem;
        margin-top: -0.6rem;
        margin-bottom: 0.8rem;
    }}
    .fx-card {{
        background: {WHITE};
        border: 1px solid {BORDER};
        border-left: 4px solid {TEAL};
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.8rem;
    }}
    .fx-card h4 {{
        color: {NAVY};
        margin: 0 0 0.35rem 0;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .fx-card p {{
        color: {SLATE};
        margin: 0;
        font-size: 0.92rem;
    }}
    .fx-value {{
        color: {NAVY};
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1.15;
    }}
    .fx-badge {{
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        color: {WHITE};
    }}
    .fx-note {{
        background: {LIGHT};
        border-left: 4px solid {SLATE};
        border-radius: 4px;
        padding: 0.7rem 0.9rem;
        font-size: 0.88rem;
        color: {SLATE};
        margin-top: 0.6rem;
    }}
    div[data-testid="stMetric"] {{
        background: {WHITE};
        border: 1px solid {BORDER};
        border-top: 3px solid {TEAL};
        border-radius: 6px;
        padding: 0.8rem 1rem;
    }}
    div[data-testid="stMetric"] label {{
        color: {SLATE} !important;
        font-size: 0.82rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: {NAVY} !important;
        font-weight: 700;
    }}
    .stButton > button {{
        background: {TEAL};
        color: {WHITE};
        border: none;
        border-radius: 5px;
        font-weight: 600;
        padding: 0.5rem 1.6rem;
    }}
    .stButton > button:hover {{
        background: {NAVY};
        color: {WHITE};
        border: none;
    }}
    footer, .stApp {{ }}
</style>
"""


def inject_css() -> None:
    """Apply the project-wide style. Safe to call on every page/reload."""
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    """Consistent page title block with teal underline."""
    st.markdown(
        f'<div class="fx-header"><h1>{title}</h1>'
        f'<div class="fx-sub">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    """Grey explanatory note box."""
    st.markdown(f'<div class="fx-note">{text}</div>', unsafe_allow_html=True)


def card(title: str, body: str) -> None:
    """White card with navy heading and teal accent."""
    st.markdown(
        f'<div class="fx-card"><h4>{title}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def risk_badge(category: str) -> None:
    """Colored risk pill (Low = teal, Moderate = amber, High = red)."""
    bg, fg = RISK_COLORS.get(category, (SLATE, WHITE))
    st.markdown(
        f'<span class="fx-badge" style="background:{bg};color:{fg}">'
        f"{category} risk</span>",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def get_metrics() -> dict[str, Any] | None:
    """Cached load of metrics.json (``None`` if training has not run)."""
    return load_metrics()


@st.cache_data(show_spinner=False)
def get_reference_stats() -> dict[str, Any] | None:
    """Cached load of reference_stats.json (form defaults / validation ranges)."""
    return load_reference_stats()


def get_bundle() -> dict[str, Any] | None:
    """Load (non-cached) model bundle; show a friendly error when missing.

    The bundle holds a fitted estimator, so it is deliberately not
    ``cache_data``-serialized; Streamlit re-imports cheaply enough and this
    avoids pickling warnings across versions.
    """
    try:
        return load_bundle()
    except ModelNotLoadedError as exc:
        st.error(
            "**Trained model not found.** Train it first, then reload this page:\n\n"
            "```bash\npython -m src.train\n```\n\n"
            f"_{exc}_"
        )
        return None


def missing_artifacts_warning(metrics: dict[str, Any] | None) -> bool:
    """Warn (once per page) when metrics.json is missing; True if missing."""
    if metrics is None:
        st.warning(
            "**Training results not found.** Run `python -m src.train` to "
            "generate `models/artifacts/metrics.json` and reload this page."
        )
        return True
    return False


def fmt_prob(value: float) -> str:
    """Format a probability without misleading trailing zeros."""
    if value < 0.0001:
        return "< 0.0001"
    if value < 0.01:
        return f"{value:.4f}"
    return f"{value:.3f}"


def fmt_metric(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def styled_df(frame: pd.DataFrame) -> None:
    """Render a dataframe full-width (compat helper for all Streamlit versions)."""
    show_dataframe(frame, hide_index=True)


def show_dataframe(frame: pd.DataFrame, **kwargs) -> None:
    """Render a dataframe stretched to the container width.

    Uses the modern ``width="stretch"`` argument and falls back to the
    deprecated ``use_container_width`` on older Streamlit releases.
    """
    try:
        st.dataframe(frame, width="stretch", **kwargs)
    except TypeError:  # Streamlit < 1.50
        st.dataframe(frame, use_container_width=True, **kwargs)


def new_fig(width: float = 7.0, height: float = 4.5):
    """Create a matplotlib figure styled with the project palette."""
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_facecolor(WHITE)
    fig.patch.set_facecolor(WHITE)
    ax.grid(alpha=0.25, linewidth=0.7, color=SLATE)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BORDER)
    ax.tick_params(colors=SLATE, labelsize=9)
    ax.xaxis.label.set_color(SLATE)
    ax.yaxis.label.set_color(SLATE)
    ax.title.set_color(NAVY)
    return fig, ax


def show_figure(fig) -> None:
    """Render and close a matplotlib figure."""
    st.pyplot(fig, clear_figure=True)


def confusion_matrix_figure(cm: list[list[int]], title: str):
    """Confusion-matrix heatmap in the project palette."""
    import numpy as np

    matrix = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Predicted\nno failure", "Predicted\nfailure"], fontsize=9)
    ax.set_yticks([0, 1], ["Actual\nno failure", "Actual\nfailure"], fontsize=9)
    thresh = matrix.max() / 2 if matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                    fontsize=15, fontweight="bold",
                    color=WHITE if matrix[i, j] > thresh else NAVY)
    ax.set_title(title, fontweight="bold", color=NAVY, fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig
