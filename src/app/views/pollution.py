"""Air-pollution page: next-day PM2.5 at the monitoring stations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app import loaders as L
from app import ui
from app.style import BLUE, GREY_LIGHT, INK, style


def _geo(lat, lon, values, selected: int | None) -> go.Scattergeo:
    colours = [L.aqi_band(v)[1] if np.isfinite(v) else "#E5E7EB" for v in values]
    labels = [f"Station {i}: {v:.0f} µg/m³ · {L.aqi_band(v)[0]}" if np.isfinite(v) else f"Station {i}: no reading" for i, v in enumerate(values)]
    sizes = [17 if i == selected else 11 for i in range(len(values))]
    lines = [3 if i == selected else 0.6 for i in range(len(values))]
    return go.Scattergeo(lat=lat, lon=lon, mode="markers", showlegend=False, text=labels, hoverinfo="text",
                         marker=dict(size=sizes, color=colours, line=dict(color=INK, width=lines)))


def render() -> None:
    ui.header(
        "Air pollution", "Next-day PM2.5",
        "An A3TGCN graph network over monitoring stations: graph convolution shares signal between nearby stations, a GRU follows time, and "
        "attention weighs the last 7 snapshots. Colours follow the US EPA PM2.5 bands.",
        badge="Demonstration", badge_kind="demo",
    )
    try:
        ctx = L.load_pollution()
    except (L.DataMissing, FileNotFoundError) as exc:
        st.error(str(exc))
        return

    ui.note(
        "The cached OpenAQ record has only a few hundred dates with a median of one reporting station per date, so snapshots are not "
        "consecutive days and most stations have no reading. This shows how the model behaves, not evidence of skill: on the 85 observed "
        "test station-days its exceedance F1 equals an always-alert rule."
    )

    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            split = ui.choice("Windows from", ["test", "val", "train"], "test", "poll_split", "Held-out test windows are the fair ones to look at.")
        idx = [i for i, s in enumerate(ctx.split) if s == split]
        if not idx:
            st.warning("No windows in this split.")
            return
        with c2:
            pos = st.select_slider(
                "Last input snapshot", options=idx, value=idx[len(idx) // 2],
                format_func=lambda i: pd.Timestamp(ctx.dates[i + ctx.window_size - 1]).strftime("%d %b %Y"),
                help="The model reads this and the 6 snapshots before it, then predicts the next snapshot.",
            )
        c3, c4 = st.columns(2)
        n_steps = c3.slider("Snapshots ahead", 1, 5, 1, help="More than 1 feeds predictions back in (not validated).")
        step = c4.slider("Show step", 1, n_steps, n_steps) if n_steps > 1 else 1
    if n_steps > 1:
        ui.note("<b>Not validated.</b> The model was trained for one step ahead; later steps feed its own predictions back in.")

    r = L.pollution_rollout(ctx, pos, n_steps)
    pred = r["pred"][step - 1]
    seen = np.isfinite(r["observed"])
    lat, lon = ctx.coords[:, 0], ctx.coords[:, 1]

    options = ["None"] + [f"Station {i}" + (" (has reading)" if seen[i] else "") for i in range(len(pred))]
    pick = st.selectbox("Highlight a station", options, help="Stations with a reading can be compared with the model.")
    sel = None if pick == "None" else options.index(pick) - 1

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scattergeo"}, {"type": "scattergeo"}]],
                        subplot_titles=(f"Model, step {step}", "Observed" if step == 1 else "Observed (step 1 only)"), horizontal_spacing=0.02)
    fig.add_trace(_geo(lat, lon, pred, sel), row=1, col=1)
    fig.add_trace(_geo(lat, lon, r["observed"] if step == 1 else np.full_like(pred, np.nan), sel), row=1, col=2)
    fig.update_geos(fitbounds="locations", showland=True, landcolor="#F3F4F6", showcountries=True, countrycolor=GREY_LIGHT,
                    showocean=True, oceancolor="#F8FAFD", coastlinecolor=GREY_LIGHT, projection_type="natural earth")
    st.plotly_chart(style(fig, 400), width="stretch")
    legend = "".join(
        f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;background:{c};margin:0 6px 0 14px;vertical-align:middle"></span>{label}'
        for _, label, c in L.AQI_BANDS
    )
    st.markdown(f'<p class="cp-foot">{legend} &nbsp;·&nbsp; grey = no reading</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Stations with a reading", f"{int(seen.sum())} of {len(seen)}")
    if seen.any() and step == 1:
        c2.metric("Model error", f"{np.sqrt(np.mean((r['pred'][0][seen] - r['observed'][seen]) ** 2)):.1f} µg/m³")
        c3.metric("Persistence error", f"{np.sqrt(np.mean((r['last_input'][seen] - r['observed'][seen]) ** 2)):.1f} µg/m³", help="Assuming tomorrow equals today.")
    else:
        c2.metric("Model error", "n/a")
        c3.metric("Persistence error", "n/a")

    if sel is not None:
        st.markdown(f"#### Station {sel}")
        days = [pd.Timestamp(ctx.dates[pos + j]).strftime("%d %b") for j in range(ctx.window_size)]
        history = ctx.windows[pos][sel, 0, :] * ctx.std + ctx.mean
        bar = go.Figure()
        bar.add_bar(x=days, y=history, name="Model input (0 = missing, set to mean)", marker_color=GREY_LIGHT)
        bar.add_bar(x=["Model next"], y=[pred[sel]], name="Model prediction", marker_color=BLUE)
        if seen[sel] and step == 1:
            bar.add_bar(x=["Observed next"], y=[r["observed"][sel]], name="Observed", marker_color=INK)
        bar.update_yaxes(title_text="PM2.5 (µg/m³)")
        st.plotly_chart(style(bar, 320), width="stretch")
    ui.footnote(f"Checkpoint: {ctx.source}. The station graph is rebuilt offline from the cached OpenAQ file.")
