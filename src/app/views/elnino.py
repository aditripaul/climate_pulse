"""El Niño page: TFT, LSTM and persistence forecasts for any held-out window."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import loaders as L
from app import ui
from app.style import BLUE, GREY, GREY_LIGHT, INK, style


def _band(times, lower, upper) -> go.Scatter:
    return go.Scatter(x=list(times) + list(times)[::-1], y=list(upper) + list(lower)[::-1], fill="toself",
                      fillcolor="rgba(47,109,181,0.15)", line=dict(width=0), name="TFT 80 % interval", hoverinfo="skip")


def _window_traces(e: dict, pos: int, show_lstm: bool, show_persist: bool) -> list:
    times = e["target_times"][pos]
    out = [_band(times, e["tft_lower"][pos], e["tft_upper"][pos]),
           go.Scatter(x=times, y=e["tft"][pos], name="TFT median", mode="lines+markers", line=dict(color=BLUE, width=2.5))]
    out.append(go.Scatter(x=times, y=e["lstm"][pos] if show_lstm else [None] * len(times), name="LSTM", mode="lines+markers",
                          line=dict(color=GREY, dash="dash"), visible=True if show_lstm else "legendonly"))
    out.append(go.Scatter(x=times, y=e["persistence"][pos] if show_persist else [None] * len(times), name="Persistence",
                          line=dict(color=GREY_LIGHT, dash="dot")))
    out.append(go.Scatter(x=times, y=e["actual"][pos], name="Observed (what happened)", mode="lines+markers", line=dict(color=INK, width=2.5)))
    return out


def _single(e: dict, pos: int, show_lstm: bool, show_persist: bool) -> go.Figure:
    times, hist = e["target_times"][pos], e["history"]
    past = hist[(hist["time"] < times[0]) & (hist["time"] >= times[0] - pd.DateOffset(months=24))]
    fig = go.Figure()
    fig.add_scatter(x=past["time"], y=past["nino34"], name="Observed (past)", line=dict(color=GREY_LIGHT, width=2))
    for t in _window_traces(e, pos, show_lstm, show_persist):
        fig.add_trace(t)
    fig.add_hline(y=0, line=dict(color=GREY_LIGHT, width=1))
    fig.add_vrect(x0=times[0], x1=times[-1], fillcolor="#F3F4F6", opacity=0.5, line_width=0, layer="below")
    fig.update_yaxes(title_text="Niño 3.4 anomaly (°C)")
    return style(fig, 440)


def _animation(e: dict) -> go.Figure:
    n = len(e["target_times"])
    first = _window_traces(e, 0, True, True)
    fig = go.Figure(first)
    fig.frames = [go.Frame(data=_window_traces(e, i, True, True), name=str(i), traces=list(range(5))) for i in range(n)]
    labels = [e["target_times"][i][0].strftime("%b %Y") for i in range(n)]
    all_t = [t for row in e["target_times"] for t in row]
    fig.update_xaxes(range=[min(all_t), max(all_t)], title_text="Month")
    lo = float(min(e["tft_lower"].min(), e["actual"].min())) - 0.2
    hi = float(max(e["tft_upper"].max(), e["actual"].max())) + 0.2
    fig.update_yaxes(range=[lo, hi], title_text="Niño 3.4 anomaly (°C)")
    fig.add_hline(y=0, line=dict(color=GREY_LIGHT, width=1))
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.28, buttons=[
            dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=350, redraw=True), fromcurrent=True)]),
            dict(label="❚❚ Pause", method="animate", args=[[None], dict(frame=dict(duration=0), mode="immediate")])])],
        sliders=[dict(x=0.14, len=0.86, y=-0.15, currentvalue=dict(prefix="Forecast issued before: "), steps=[
            dict(method="animate", label=labels[i], args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for i in range(n)])],
    )
    return style(fig, 500)


def render() -> None:
    ui.header(
        "El Niño", "12-month forecasts",
        "Niño 3.4 sea-surface temperature anomaly. The Temporal Fusion Transformer reads 60 months of history and forecasts each of the "
        "next 12 months with a 10–90 % interval; an LSTM and simple persistence are shown for comparison.",
        badge="Validated on held-out years",
    )
    try:
        e = L.load_elnino()
    except (L.DataMissing, FileNotFoundError) as exc:
        st.error(str(exc))
        return

    tab_pick, tab_play = st.tabs([":material/tune: Explore a forecast", ":material/play_circle: Watch it unfold"])
    with tab_pick:
        starts = [t[0] for t in e["target_times"]]
        with st.container(border=True):
            pos = st.select_slider(
                "Forecast issued just before", options=list(range(len(starts))), value=len(starts) // 2,
                format_func=lambda i: starts[i].strftime("%B %Y"),
                help="Each position is a forecast made from the 60 months before this date, scored on the 12 months after.",
            )
            c1, c2, _ = st.columns([1, 1, 3])
            show_lstm = c1.toggle("LSTM", value=True)
            show_persist = c2.toggle("Persistence", value=True)
        st.plotly_chart(_single(e, pos, show_lstm, show_persist), width="stretch")

        def err(key: str) -> float:
            return float(np.sqrt(np.mean((e[key][pos] - e["actual"][pos]) ** 2)))

        inside = float(np.mean((e["actual"][pos] >= e["tft_lower"][pos]) & (e["actual"][pos] <= e["tft_upper"][pos])))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TFT error", f"{err('tft'):.2f} °C", help="RMSE over the 12 months of this window.")
        c2.metric("LSTM error", f"{err('lstm'):.2f} °C")
        c3.metric("Persistence error", f"{err('persistence'):.2f} °C", help="Assuming the anomaly stays at its last value.")
        c4.metric("Reality inside interval", f"{inside:.0%}", help="Share of the 12 months that fell inside the 10–90 % band.")
    with tab_play:
        st.caption("Press Play to sweep through the whole 2015–2026 test period and watch the forecasts chase reality.")
        st.plotly_chart(_animation(e), width="stretch")

    st.markdown("#### Across the whole test period")
    m = e["metrics"]
    leads = list(range(1, len(m["lead_rmse"]["tft"]) + 1))
    fig = go.Figure()
    fig.add_scatter(x=leads, y=m["lead_rmse"]["tft"], name="TFT", mode="lines+markers", line=dict(color=BLUE, width=3))
    fig.add_scatter(x=leads, y=m["lead_rmse"]["lstm"], name="LSTM", mode="lines+markers", line=dict(color=GREY, dash="dash"))
    fig.add_scatter(x=leads, y=m["lead_rmse"]["persistence"], name="Persistence", mode="lines+markers", line=dict(color=GREY_LIGHT, dash="dot"))
    fig.update_xaxes(title_text="Lead time (months)", dtick=1)
    fig.update_yaxes(title_text="RMSE (°C)")
    left, right = st.columns([3, 2])
    left.plotly_chart(style(fig, 330, "Error by lead time"), width="stretch")
    right.dataframe(
        pd.DataFrame({
            "RMSE (°C)": [m[k]["rmse"] for k in ("tft", "lstm", "persistence")],
            "ACC": [m[k]["acc"] for k in ("tft", "lstm", "persistence")],
            "Skill": [m[k]["skill_score"] for k in ("tft", "lstm", "persistence")],
            "Bias (°C)": [m[k]["bias"] for k in ("tft", "lstm", "persistence")],
        }, index=["TFT", "LSTM", "Persistence"]).round(3),
        width="stretch",
    )
    ui.footnote(
        f"Checkpoints: {e['source']}. Single-run (seed 42) figures; across five seeds the TFT averages RMSE 0.78 ± 0.08 °C, so its edge over "
        "the LSTM is within run-to-run variation. Both models run about 0.3 °C cold in the warmer test period."
    )
