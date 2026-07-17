"""
Color palettes and theming for ClimatePulse visualizations.

Provides a unified design system for all charts across the
dashboard and notebooks.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────
# Primary palette
# ──────────────────────────────────────────────────────────────
PRIMARY = "#00D4AA"  # Teal-green
ACCENT = "#7C3AED"  # Violet
WARNING_COLOR = "#F59E0B"  # Amber
DANGER = "#EF4444"  # Red
INFO = "#3B82F6"  # Blue

# ──────────────────────────────────────────────────────────────
# Dark-mode surface colours
# ──────────────────────────────────────────────────────────────
BG_DARK = "#0A0E1A"
SURFACE = "#141B2D"
GLASS_BG = "rgba(17, 25, 40, 0.75)"
GLASS_BORDER = "rgba(255, 255, 255, 0.125)"
TEXT_PRIMARY = "#E2E8F0"
TEXT_SECONDARY = "#94A3B8"

# ──────────────────────────────────────────────────────────────
# Climate-specific palettes
# ──────────────────────────────────────────────────────────────

# Diverging blue → red for temperature anomalies
TEMPERATURE_SCALE = [
    "#08306b",
    "#2171b5",
    "#4292c6",
    "#6baed6",
    "#9ecae1",
    "#c6dbef",
    "#fcbba1",
    "#fb6a4a",
    "#ef3b2c",
    "#cb181d",
    "#67000d",
]

# Sequential teal for SST
SST_SCALE = ["#f7fcfd", "#e5f5f9", "#ccece6", "#99d8c9", "#66c2a4", "#41ae76", "#238b45", "#005824"]

# AQI categorical colours (US EPA standard)
AQI_COLORS = {
    "Good": "#00E400",
    "Moderate": "#FFFF00",
    "Unhealthy-Sensitive": "#FF7E00",
    "Unhealthy": "#FF0000",
    "Very Unhealthy": "#8F3F97",
    "Hazardous": "#7E0023",
}

# Emissions by gas
GAS_COLORS = {
    "co2": "#EF4444",
    "ch4": "#F59E0B",
    "n2o": "#8B5CF6",
    "co2e_100yr": "#EC4899",
}

# Model comparison
MODEL_COLORS = {
    "TFT": PRIMARY,
    "LSTM": ACCENT,
    "Bayesian": "#F59E0B",
    "OLS": "#6B7280",
    "Persistence": "#374151",
}


# ──────────────────────────────────────────────────────────────
# Plotly theme
# ──────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY, size=13),
    title_font=dict(family="Outfit, sans-serif", size=20, color=TEXT_PRIMARY),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.1)",
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.1)",
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_SECONDARY),
    ),
    margin=dict(l=40, r=20, t=50, b=40),
    hoverlabel=dict(
        bgcolor=SURFACE,
        font_color=TEXT_PRIMARY,
        bordercolor=GLASS_BORDER,
    ),
)


def apply_theme(fig):
    """Apply the ClimatePulse dark theme to a Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig
