"""Warming-rate page: interactive change-point analysis of GISTEMP."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import loaders as L
from app import ui
from app.style import BLUE, GREY, GREY_LIGHT, INK, style


def render() -> None:
    ui.header(
        "Warming rate", "When did warming change speed?",
        "A continuous piecewise-linear trend with autocorrelated (AR(1)) errors is fitted to annual global-mean temperature anomalies, and BIC "
        "chooses how many breaks the data justify. No neural network is involved.",
        badge="Statistical model",
    )
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1, 1])
        max_k = c1.slider("Most breaks to consider", 0, 3, 3, help="0 forces a single straight line.")
        min_seg = c2.slider("Shortest segment (years)", 5, 20, 10, help="Stops the search from fitting very short segments.")
        straight = c3.toggle("Compare with a straight line", value=True)
    try:
        res = L.changepoints(max_k, min_seg)
    except (L.DataMissing, FileNotFoundError) as exc:
        st.error(str(exc))
        return

    annual = res["annual"]
    fig = go.Figure()
    fig.add_scatter(x=annual["year"], y=annual["global_temp_anomaly"], mode="markers", name="Annual mean", marker=dict(color=INK, size=7),
                    hovertemplate="%{x}: %{y:+.2f} °C<extra></extra>")
    if straight:
        slope, icpt = np.polyfit(annual["year"], annual["global_temp_anomaly"], 1)
        fig.add_scatter(x=annual["year"], y=slope * annual["year"] + icpt, name=f"Straight line ({10 * slope:.3f} °C/decade)",
                        line=dict(color=GREY_LIGHT, dash="dash"))
    fig.add_scatter(x=annual["year"], y=res["fitted"], name="Selected trend", line=dict(color=BLUE, width=3.5))
    for cp in res["changepoints"]:
        fig.add_vline(x=cp["time"], line=dict(color=GREY, dash="dot"), annotation_text=f"{cp['time']:.0f}", annotation_position="top")
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Global temperature anomaly (°C)")
    st.plotly_chart(style(fig, 440), width="stretch")

    slopes = [10 * s for s in res["segment_slopes"]]
    edges = [int(annual["year"].min())] + [int(cp["time"]) for cp in res["changepoints"]] + [int(annual["year"].max())]
    c1, c2, c3 = st.columns(3)
    c1.metric("Breaks selected", f"{res['n_changepoints']}")
    c2.metric("Warming now", f"{slopes[-1]:.3f} °C / decade")
    c3.metric("Warming at the start", f"{slopes[0]:.3f} °C / decade")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Segments")
        st.dataframe(pd.DataFrame({"From": edges[:-1], "To": edges[1:], "Warming (°C / decade)": slopes}).round(3), hide_index=True, width="stretch")
    with right:
        st.markdown("#### Model selection (lower BIC is better)")
        bic = pd.DataFrame({"Breaks": list(res["bic_by_k"]), "BIC": list(res["bic_by_k"].values())})
        bic["Δ vs best"] = bic["BIC"] - bic["BIC"].min()
        st.dataframe(bic.round(2), hide_index=True, width="stretch")
    ui.footnote(
        f"AR(1) coefficient {res['ar1_phi']:.2f}. The reported result is one break in 1976 (0.035 → 0.211 °C/decade); a second break around 2012 "
        "is nearly as well supported but not established."
    )
