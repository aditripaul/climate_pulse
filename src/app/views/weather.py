"""Weather model page: animated global forecast against ERA5, as maps and as a 3D globe."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from app import loaders as L
from app import ui
from app.style import BLUE, DIVERGING, GREY_LIGHT, SEQUENTIAL, style


def _fields(r: dict, field: str) -> tuple[np.ndarray, np.ndarray, str, float, float]:
    if field == "Temperature":
        truth, pred, unit = r["truth_k"], r["pred_k"], "K"
        lo, hi = float(truth.min()), float(truth.max())
    else:
        truth, pred, unit = r["truth_mm"], r["pred_mm"], "mm / 6 h"
        lo, hi = 0.0, float(np.percentile(truth, 99.5)) or 1.0
    return truth, pred, unit, lo, hi


def _play_controls(fig: go.Figure, labels: list[str], y_button: float, y_slider: float) -> None:
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=y_button, buttons=[
            dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=550, redraw=True), fromcurrent=True)]),
            dict(label="❚❚ Pause", method="animate", args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
        ])],
        sliders=[dict(x=0.14, len=0.86, y=y_slider, active=0, currentvalue=dict(prefix="Lead: ", visible=True), steps=[
            dict(method="animate", label=label, args=[[str(i)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
            for i, label in enumerate(labels)])],
    )


def _maps(r: dict, field: str) -> go.Figure:
    truth, pred, unit, lo, hi = _fields(r, field)
    err = pred - truth
    emax = float(np.abs(err).max()) or 1.0
    lon, lat, n = r["lon"], r["lat"], len(r["times"])

    def traces(i: int) -> list[go.Heatmap]:
        return [
            go.Heatmap(z=truth[i], x=lon, y=lat, coloraxis="coloraxis", hovertemplate="lon %{x}°, lat %{y}°<br>%{z:.1f}<extra>ERA5</extra>"),
            go.Heatmap(z=pred[i], x=lon, y=lat, coloraxis="coloraxis", hovertemplate="lon %{x}°, lat %{y}°<br>%{z:.1f}<extra>Model</extra>"),
            go.Heatmap(z=err[i], x=lon, y=lat, coloraxis="coloraxis2", hovertemplate="lon %{x}°, lat %{y}°<br>%{z:+.2f}<extra>Error</extra>"),
        ]

    fig = make_subplots(rows=1, cols=3, subplot_titles=("ERA5 (truth)", "Model", "Error (model − ERA5)"), horizontal_spacing=0.075)
    for c, t in enumerate(traces(0), start=1):
        fig.add_trace(t, row=1, col=c)
    fig.update_layout(
        coloraxis=dict(colorscale=SEQUENTIAL, cmin=lo, cmax=hi, colorbar=dict(title=unit, x=0.665, len=0.8, thickness=12)),
        coloraxis2=dict(colorscale=DIVERGING, cmin=-emax, cmax=emax, colorbar=dict(title=unit, x=1.03, len=0.8, thickness=12)),
    )
    fig.update_xaxes(title_text="Longitude (°)")
    fig.update_yaxes(title_text="Latitude (°)", col=1)
    fig.update_yaxes(showticklabels=False, col=2)
    fig.update_yaxes(showticklabels=False, col=3)
    if n > 1:
        fig.frames = [go.Frame(data=traces(i), name=str(i), traces=[0, 1, 2]) for i in range(n)]
        _play_controls(fig, [f"+{6 * (i + 1)} h" for i in range(n)], -0.34, -0.2)
    style(fig, height=560 if n > 1 else 400)
    fig.update_layout(margin=dict(l=50, r=90, t=60, b=140 if n > 1 else 60))
    return fig


def _globe(r: dict, field: str, show: str) -> go.Figure:
    truth, pred, unit, lo, hi = _fields(r, field)
    data = {"Model": pred, "ERA5 (truth)": truth, "Error": pred - truth}[show]
    if show == "Error":
        emax = float(np.abs(data).max()) or 1.0
        scale, lo, hi = DIVERGING, -emax, emax
    else:
        scale = SEQUENTIAL
    lat, lon = np.radians(r["lat"]), np.radians(np.append(r["lon"], 360.0))
    x = np.cos(lat)[:, None] * np.cos(lon)[None, :]
    y = np.cos(lat)[:, None] * np.sin(lon)[None, :]
    z = np.repeat(np.sin(lat)[:, None], len(lon), axis=1)

    def seam(a: np.ndarray) -> np.ndarray:  # close the longitude seam
        return np.concatenate([a, a[:, :1]], axis=1)

    def surface(i: int) -> go.Surface:
        return go.Surface(x=x, y=y, z=z, surfacecolor=seam(data[i]), colorscale=scale, cmin=lo, cmax=hi,
                          colorbar=dict(title=unit, thickness=12, len=0.7), showscale=True, lighting=dict(ambient=0.9, diffuse=0.3),
                          hovertemplate=f"%{{surfacecolor:.1f}} {unit}<extra>{show}</extra>")

    n = len(r["times"])
    fig = go.Figure(surface(0))
    if n > 1:
        fig.frames = [go.Frame(data=[surface(i)], name=str(i), traces=[0]) for i in range(n)]
        _play_controls(fig, [f"+{6 * (i + 1)} h" for i in range(n)], -0.02, 0.0)
    ax = dict(visible=False)
    fig.update_layout(scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="data", camera=dict(eye=dict(x=1.5, y=0.7, z=0.6))))
    style(fig, height=560)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    return fig


def render() -> None:
    ui.header(
        "Neural weather model", "Next 6 hours, globally",
        "A graph network on a 2,000-node Fibonacci mesh, trained on ERA5 2020–2021. It maps the two latest 6-hourly fields "
        "of temperature and precipitation to the next one. Pick a moment from the held-out test period and compare with reality.",
        badge="Validated for one 6 h step",
    )

    with st.container(border=True):
        c1, c2, c3 = st.columns([2.2, 1.6, 1.6])
        variant = c1.selectbox("Model version", list(L.NWM_VARIANTS), help="The main model and three ablation variants trained for the paper.")
        with c2:
            mode = ui.choice("Mode", ["Next 6 h", "Roll forward"], "Next 6 h", "mode", "Roll forward feeds the model its own output.")
        with c3:
            field = ui.choice("Field", ["Temperature", "Precipitation"], "Temperature", "field")
        try:
            ctx = L.load_nwm(variant)
        except (L.DataMissing, FileNotFoundError) as exc:
            st.error(str(exc))
            return
        times = L.test_times(ctx)
        pos = st.select_slider(
            "Start time (held-out test period)", options=list(range(len(times))), value=len(times) // 2,
            format_func=lambda i: times[i].strftime("%d %b %Y, %H:%M UTC"),
            help="The model sees the two fields before this time and predicts the field at this time.",
        )
        n_steps = 1
        if mode == "Roll forward":
            max_steps = len(ctx.dataset.times) - ctx.dataset.valid_indices[pos]
            n_steps = st.slider("Steps ahead (6 h each)", 2, int(min(24, max(2, max_steps))), min(8, int(max_steps)),
                                help="Up to 6 days. Press Play on the chart to animate.")
        nino_override = None
        if ctx.conditioning != "none":
            with st.expander("Advanced: change the El Niño input"):
                if st.toggle("Override Niño 3.4 with a constant"):
                    nino_override = st.slider("Niño 3.4 anomaly (°C)", -3.0, 3.0, 0.0, 0.1)
                    st.caption("The ablation found this input has no measurable effect on 6-hour forecasts. Try both extremes and watch the metrics.")

    if n_steps > 1:
        ui.note(
            "<b>Not validated.</b> The model was trained and tested only for a single 6-hour step. Feeding its own predictions "
            "back in is outside that: expect errors to grow and fields to drift. This is not forecast skill."
        )

    with st.spinner("Running the model..."):
        r = L.nwm_rollout(ctx, pos, n_steps, nino_override)

    m1, m2, m3 = st.columns(3)
    m1.metric("Model error at last step", f"{r['rmse_k'][-1]:.2f} K", help="Area-weighted RMSE against ERA5.")
    m2.metric("Persistence error", f"{r['persistence_rmse_k'][-1]:.2f} K", help="Assuming the weather stays as it was.")
    m3.metric("Niño 3.4 input", f"{r['nino'][0]:+.2f} °C")

    tab_maps, tab_globe = st.tabs([":material/map: Maps", ":material/public: Globe"])
    with tab_maps:
        st.plotly_chart(_maps(r, field), width="stretch")
    with tab_globe:
        show = ui.choice("Show", ["Model", "ERA5 (truth)", "Error"], "Model", "globe_show", "Drag to rotate, scroll to zoom.")
        st.plotly_chart(_globe(r, field, show), width="stretch")

    if n_steps > 1:
        lead = [6 * (i + 1) for i in range(n_steps)]
        fig = go.Figure()
        fig.add_scatter(x=lead, y=r["rmse_k"], name="Model (rolled forward)", mode="lines+markers", line=dict(color=BLUE, width=3))
        fig.add_scatter(x=lead, y=r["persistence_rmse_k"], name="Persistence", mode="lines+markers", line=dict(color=GREY_LIGHT, dash="dash"))
        fig.update_xaxes(title_text="Lead time (hours)")
        fig.update_yaxes(title_text="Area-weighted RMSE (K)")
        st.plotly_chart(style(fig, 300, "Error against ERA5 as the rollout continues"), width="stretch")

    with st.expander("Reproduce the reported test score"):
        st.write("Scores every held-out sample with this checkpoint in batches of 16, as in the paper (area-weighted 2 m temperature RMSE).")
        if st.button("Run evaluation on the full test period", type="primary"):
            from models.nwm.evaluate import evaluate_nwm_checkpoint

            with st.spinner("Scoring 438 samples..."):
                m = evaluate_nwm_checkpoint(ctx.checkpoint, ctx.conditioning)
            st.success(f"Model {m['t2m_rmse_K']:.3f} K vs persistence {m['persistence_t2m_rmse_K']:.3f} K over {m['n_test']} samples.")
    ui.footnote(
        f"Checkpoint: {ctx.checkpoint} ({ctx.source}). Predictions here are computed one sample at a time. The model's graph LayerNorm pools "
        "statistics across a batch, so scores differ slightly with batch size (about 1.217 K one at a time against 1.214 K in batches of 16)."
    )
