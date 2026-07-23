"""
Chart builders for ClimatePulse.

Reusable Plotly chart factories for the dashboard and notebooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualization.themes import (
    ACCENT,
    AQI_COLORS,
    GAS_COLORS,
    MODEL_COLORS,
    PRIMARY,
    TEMPERATURE_SCALE,
    TEXT_SECONDARY,
    apply_theme,
)


def forecast_chart(
    historical: pd.DataFrame,
    predictions: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    time_col: str = "time",
    value_col: str = "nino34",
    title: str = "El Niño 3.4 Forecast",
) -> go.Figure:
    """
    Create an El Niño forecast chart with confidence bands.
    """
    fig = go.Figure()

    # Historical
    fig.add_trace(
        go.Scatter(
            x=historical[time_col],
            y=historical[value_col],
            mode="lines",
            name="Observed",
            line=dict(color=PRIMARY, width=2),
        )
    )

    # Forecast timeline
    last_date = pd.to_datetime(historical[time_col].iloc[-1])
    forecast_dates = pd.date_range(
        last_date + pd.DateOffset(months=1),
        periods=len(predictions),
        freq="MS",
    )

    # Confidence band
    if lower is not None and upper is not None:
        fig.add_trace(
            go.Scatter(
                x=list(forecast_dates) + list(forecast_dates[::-1]),
                y=list(upper) + list(lower[::-1]),
                fill="toself",
                fillcolor="rgba(0, 212, 170, 0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="80% Confidence",
                showlegend=True,
            )
        )

    # Forecast line
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=predictions,
            mode="lines+markers",
            name="Forecast",
            line=dict(color=ACCENT, width=2.5, dash="dot"),
            marker=dict(size=5),
        )
    )

    # El Niño / La Niña threshold lines
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="rgba(239,68,68,0.4)",
        annotation_text="El Niño threshold",
    )
    fig.add_hline(
        y=-0.5,
        line_dash="dash",
        line_color="rgba(59,130,246,0.4)",
        annotation_text="La Niña threshold",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="SST Anomaly (°C)",
    )
    return apply_theme(fig)


def warming_stripes(
    df: pd.DataFrame,
    time_col: str = "time",
    value_col: str = "global_temp_anomaly",
    title: str = "Global Temperature Anomaly (Warming Stripes)",
) -> go.Figure:
    """
    Create an Ed Hawkins-style warming stripes visualization.
    """
    df = df.copy()
    df["year"] = pd.to_datetime(df[time_col]).dt.year
    annual = df.groupby("year")[value_col].mean().reset_index()

    fig = go.Figure()

    vmin = annual[value_col].min()
    vmax = annual[value_col].max()

    for _, row in annual.iterrows():
        normalized = (row[value_col] - vmin) / (vmax - vmin + 1e-8)
        idx = int(normalized * (len(TEMPERATURE_SCALE) - 1))
        idx = max(0, min(idx, len(TEMPERATURE_SCALE) - 1))
        color = TEMPERATURE_SCALE[idx]

        fig.add_trace(
            go.Bar(
                x=[row["year"]],
                y=[1],
                marker_color=color,
                showlegend=False,
                hovertemplate=f"Year: {int(row['year'])}<br>Anomaly: {row[value_col]:.2f}°C<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        barmode="stack",
        bargap=0,
        xaxis=dict(title="Year", showgrid=False),
        yaxis=dict(visible=False),
        height=200,
    )
    return apply_theme(fig)


def emissions_temperature_chart(
    df: pd.DataFrame,
    title: str = "Emissions vs Temperature Anomaly",
) -> go.Figure:
    """
    Dual-axis chart showing emissions and temperature over time.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Temperature on primary y-axis
    if "global_temp_anomaly" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["year"],
                y=df["global_temp_anomaly"],
                name="Temperature Anomaly (°C)",
                line=dict(color="#EF4444", width=2.5),
            ),
            secondary_y=False,
        )

    # Emissions on secondary y-axis
    emission_cols = [c for c in df.columns if c in GAS_COLORS]
    for col in emission_cols:
        fig.add_trace(
            go.Scatter(
                x=df["year"],
                y=df[col],
                name=f"{col.upper()} Emissions",
                line=dict(color=GAS_COLORS.get(col, ACCENT), width=1.5, dash="dot"),
            ),
            secondary_y=True,
        )

    fig.update_layout(title=title)
    fig.update_yaxes(title_text="Temperature Anomaly (°C)", secondary_y=False)
    fig.update_yaxes(title_text="Emissions (tonnes)", secondary_y=True)
    return apply_theme(fig)


def model_comparison_chart(
    metrics_df: pd.DataFrame,
    title: str = "Model Comparison",
) -> go.Figure:
    """
    Bar chart comparing model metrics side by side.
    """
    fig = go.Figure()

    metric_cols = [c for c in metrics_df.columns if c != "model"]
    models = metrics_df.index.tolist()

    for model_name in models:
        color = MODEL_COLORS.get(model_name, PRIMARY)
        fig.add_trace(
            go.Bar(
                name=model_name,
                x=metric_cols,
                y=metrics_df.loc[model_name, metric_cols].values,
                marker_color=color,
            )
        )

    fig.update_layout(
        title=title,
        barmode="group",
        xaxis_title="Metric",
        yaxis_title="Value",
    )
    return apply_theme(fig)



def aqi_gauge(value: float, title: str = "AQI") -> go.Figure:
    """Create a radial gauge for AQI display."""
    if value <= 50:
        color = AQI_COLORS["Good"]
    elif value <= 100:
        color = AQI_COLORS["Moderate"]
    elif value <= 150:
        color = AQI_COLORS["Unhealthy-Sensitive"]
    elif value <= 200:
        color = AQI_COLORS["Unhealthy"]
    elif value <= 300:
        color = AQI_COLORS["Very Unhealthy"]
    else:
        color = AQI_COLORS["Hazardous"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title=dict(text=title, font=dict(color=TEXT_SECONDARY, size=14)),
            number=dict(font=dict(size=36, color=color)),
            gauge=dict(
                axis=dict(range=[0, 500], tickcolor=TEXT_SECONDARY),
                bar=dict(color=color),
                bgcolor="rgba(0,0,0,0)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 50], color="rgba(0,228,0,0.15)"),
                    dict(range=[50, 100], color="rgba(255,255,0,0.15)"),
                    dict(range=[100, 150], color="rgba(255,126,0,0.15)"),
                    dict(range=[150, 200], color="rgba(255,0,0,0.15)"),
                    dict(range=[200, 300], color="rgba(143,63,151,0.15)"),
                    dict(range=[300, 500], color="rgba(126,0,35,0.15)"),
                ],
            ),
        )
    )

    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))
    return apply_theme(fig)
