"""Shared helpers for the Streamlit dashboard.

Provides theme-adaptive styling and error-tolerant loading for model artifacts.
The dashboard follows Streamlit's built-in Light/Dark theme setting.
"""

from __future__ import annotations

from html import escape
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

# Industrial accent palette
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

# CSS variables inherit the active Streamlit theme.
CSS = """
<style>
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    .block-container {
        max-width: 1140px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    header[data-testid="stHeader"] {
        background: var(--background-color);
    }

    /* Keep the sidebar consistently navy in both themes. */
    [data-testid="stSidebar"] {
        background-color: #0f2a4a;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
        color: #e2e8f0 !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }

    /* Theme-aware headings and body text. */
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6 {
        color: var(--text-color) !important;
        font-family: "Source Sans Pro", sans-serif;
        letter-spacing: -0.01em;
    }

    .stApp p,
    .stApp li,
    .stApp label {
        color: var(--text-color);
    }

    /* Page header */
    .fx-header {
        border-bottom: 3px solid #0d9488;
        padding-bottom: 0.4rem;
        margin-bottom: 1.2rem;
    }

    .fx-header h1 {
        color: var(--text-color) !important;
        margin-bottom: 0.8rem;
    }

    .fx-sub {
        color: var(--text-color) !important;
        opacity: 0.85;
        font-size: 0.95rem;
        margin-top: -0.6rem;
        margin-bottom: 0.8rem;
    }

    /* Information cards */
    .fx-card {
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid var(--secondary-background-color);
        border-left: 4px solid #0d9488;
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.8rem;
    }

    .fx-card h4 {
        color: var(--text-color) !important;
        margin: 0 0 0.35rem 0;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .fx-card p {
        color: var(--text-color) !important;
        margin: 0;
        font-size: 0.92rem;
    }

    .fx-value {
        color: var(--text-color) !important;
        font-size: 1.9rem;
        font-weight: 700;
        line-height: 1.15;
    }

    /* Risk badges keep their own high-contrast colors. */
    .fx-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        color: #ffffff !important;
    }

    /* Explanatory notes */
    .fx-note {
        background: var(--secondary-background-color);
        border-left: 4px solid #0d9488;
        border-radius: 4px;
        padding: 0.7rem 0.9rem;
        font-size: 0.88rem;
        color: var(--text-color) !important;
        margin-top: 0.6rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid var(--secondary-background-color);
        border-top: 3px solid #0d9488;
        border-radius: 6px;
        padding: 0.8rem 1rem;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: var(--text-color) !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-color) !important;
        font-weight: 700;
    }

    /* Buttons */
    .stButton > button {
        background: #0d9488;
        color: #ffffff !important;
        border: 1px solid #0d9488;
        border-radius: 5px;
        font-weight: 600;
        padding: 0.5rem 1.6rem;
    }

    .stButton > button:hover {
        background: #115e59;
        color: #ffffff !important;
        border-color: #115e59;
    }

    /* Inputs retain Streamlit's theme-controlled backgrounds. */
    .stApp input,
    .stApp textarea {
        color: var(--text-color);
    }

    /* Links and captions */
    .stApp a {
        color: #0d9488;
    }

    .stApp [data-testid="stCaptionContainer"] {
        color: var(--text-color);
        opacity: 0.8;
    }

    /* Table containers */
    .stApp [data-testid="stDataFrame"],
    .stApp [data-testid="stTable"] {
        border: 1px solid var(--secondary-background-color);
        border-radius: 6px;
    }
</style>
"""


def inject_css() -> None:
    """Apply theme-adaptive styling without adding another theme selector."""
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    """Render a consistent page heading and subtitle."""
    st.markdown(
        f"""
        <div class="fx-header">
            <h1>{escape(str(title))}</h1>
        </div>
        <div class="fx-sub">{escape(str(subtitle))}</div>
        """,
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    """Render an explanatory note."""
    st.markdown(
        f'<div class="fx-note">{escape(str(text))}</div>',
        unsafe_allow_html=True,
    )


def card(title: str, body: str) -> None:
    """Render a styled information card."""
    st.markdown(
        f"""
        <div class="fx-card">
            <h4>{escape(str(title))}</h4>
            <p>{escape(str(body))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(category: str) -> None:
    """Render a badge for a risk category."""
    background, foreground = RISK_COLORS.get(category, (SLATE, WHITE))

    st.markdown(
        f"""
        <span class="fx-badge"
              style="background-color: {background};
                     color: {foreground};">
            {escape(str(category))}
        </span>
        """,
        unsafe_allow_html=True,
    )


def get_metrics() -> Any:
    """Load model evaluation metrics."""
    try:
        return load_metrics()
    except (FileNotFoundError, ModelNotLoadedError):
        return None


def get_reference_stats() -> Any:
    """Load reference dataset statistics."""
    try:
        return load_reference_stats()
    except (FileNotFoundError, ModelNotLoadedError):
        return None


def get_bundle() -> Any:
    """Load the trained model bundle."""
    try:
        return load_bundle()
    except (FileNotFoundError, ModelNotLoadedError):
        return None


def missing_artifacts_warning(metrics: Any = None) -> bool:
    """Warn when model artifacts or evaluation metrics are unavailable.

    Returns:
        True if metrics are unavailable and the warning is displayed.
        False if metrics loaded successfully.
    """
    if metrics is not None:
        return False

    st.warning(
        "Model artifacts or evaluation metrics could not be loaded. "
        "Please check that the trained model files are available."
    )
    return True


def fmt_prob(value: Any) -> str:
    """Format a probability as a percentage."""
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "N/A"


def fmt_metric(value: Any, digits: int = 3) -> str:
    """Format a numeric evaluation metric."""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "N/A"


def styled_df(df: pd.DataFrame) -> Any:
    """Return a styled dataframe."""
    return df.style


def show_dataframe(df: pd.DataFrame, **kwargs: Any) -> None:
    """Display a dataframe."""
    st.dataframe(df, use_container_width=True, **kwargs)


def _is_dark_theme() -> bool:
    """Read the configured Streamlit base theme for Matplotlib charts."""
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return False


def new_fig(
    *args: Any,
    figsize: tuple[float, float] = (8, 4),
) -> tuple[Any, Any]:
    """Create a theme-aware Matplotlib figure.

    Supports new_fig(6.2, 3.6), new_fig((6.2, 3.6)),
    and new_fig(figsize=(6.2, 3.6)).
    """
    if len(args) == 2:
        figsize = (float(args[0]), float(args[1]))
    elif len(args) == 1:
        figsize = args[0]
    elif len(args) > 2:
        raise TypeError("new_fig accepts at most two positional arguments")

    dark = _is_dark_theme()

    background = "#0e1117" if dark else "#ffffff"
    foreground = "#fafafa" if dark else "#0f2a4a"
    grid_color = "#475569" if dark else "#e2e8f0"

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    ax.tick_params(colors=foreground)
    ax.xaxis.label.set_color(foreground)
    ax.yaxis.label.set_color(foreground)
    ax.title.set_color(foreground)

    for spine in ax.spines.values():
        spine.set_color(grid_color)

    ax.grid(axis="y", color=grid_color, alpha=0.6)
    fig.tight_layout()

    return fig, ax


def show_figure(fig: Any) -> None:
    """Display and close a Matplotlib figure."""
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def confusion_matrix_figure(
    confusion_matrix: Any,
    labels: list[str] | None = None,
) -> Any:
    """Create a theme-aware confusion matrix figure."""
    from sklearn.metrics import ConfusionMatrixDisplay

    dark = _is_dark_theme()
    foreground = "#fafafa" if dark else "#0f2a4a"
    background = "#0e1117" if dark else "#ffffff"

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    display_labels = labels if labels is not None else ["No Failure", "Failure"]

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=display_labels,
    )
    display.plot(ax=ax, colorbar=False, cmap="Blues")

    ax.set_title("Confusion Matrix", color=foreground)
    ax.tick_params(colors=foreground)
    ax.xaxis.label.set_color(foreground)
    ax.yaxis.label.set_color(foreground)

    for spine in ax.spines.values():
        spine.set_color(foreground)

    for text_item in ax.texts:
        text_item.set_color("#111827")

    fig.tight_layout()
    return fig