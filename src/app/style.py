"""Shared colours and Plotly layout helpers (white and grey with a blue accent)."""

from __future__ import annotations

import plotly.graph_objects as go

BLUE = "#2F6DB5"
BLUE_DARK = "#1E4E8C"
BLUE_LIGHT = "#A9C4E8"
INK = "#1F2937"
GREY_DARK = "#4B5563"
GREY = "#6B7280"
GREY_LIGHT = "#9CA3AF"
GRID = "#E5E7EB"

SEQUENTIAL = [[0.0, "#F3F6FB"], [0.5, "#7FA7D9"], [1.0, "#13294B"]]
DIVERGING = [[0.0, GREY_DARK], [0.5, "#FFFFFF"], [1.0, BLUE]]


def style(fig: go.Figure, height: int = 420, title: str | None = None) -> go.Figure:
    """Apply the white/grey/blue look to a figure."""
    fig.update_layout(
        height=height,
        title=dict(text=title, x=0.0, font=dict(size=16, color=INK)) if title else None,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Source Sans Pro, Arial, sans-serif", color=INK, size=13),
        margin=dict(l=50, r=30, t=60 if title else 30, b=80),
        legend=dict(orientation="h", yanchor="top", y=-0.24, x=0),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=GREY_LIGHT, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=GREY_LIGHT, zeroline=False)
    return fig
