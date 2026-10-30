"""
ClimatePulse simulation app.

Run with ``make app``. Uses the already-trained checkpoints; nothing is trained here.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `config`, `models`, `data` and `app` importable when launched by Streamlit
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from app import loaders as L  # noqa: E402
from app import ui  # noqa: E402
from app.views import elnino, pollution, warming, weather  # noqa: E402

MODELS = [
    ("Weather model", "nwm/nwm_best.pt"),
    ("El Niño TFT", "el_nino/tft"),
    ("El Niño LSTM", "el_nino/lstm/lstm_best.pt"),
    ("Air pollution A3TGCN", "pollution/stgnn_best.pt"),
]


def overview() -> None:
    st.markdown(
        '<div class="cp-hero"><div class="cp-eyebrow">Interactive simulations</div><h1>ClimatePulse</h1>'
        "<p>Run the trained models on held-out data and see what they get right and where they fail. "
        "Pick a simulation from the tabs above.</p></div>",
        unsafe_allow_html=True,
    )
    cards = [
        ("Weather", "Weather model", ":material/public:", "Global 2 m temperature and rain, 6 hours ahead, animated against ERA5 on maps and a 3D globe.", "Validated for one step"),
        ("El Niño", "El Niño", ":material/waves:", "12-month Niño 3.4 forecasts with uncertainty bands from the TFT, compared with an LSTM.", "Validated"),
        ("Air pollution", "Air pollution", ":material/air:", "Next-day PM2.5 at monitoring stations on a map, coloured by air-quality band.", "Demonstration"),
        ("Warming rate", "Warming rate", ":material/trending_up:", "Find when global warming changed speed, with adjustable break search.", "Statistical model"),
    ]
    cols = st.columns(4)
    for col, (page, title, _icon, text, tag) in zip(cols, cards, strict=True):
        with col, st.container(border=True):
            st.markdown(f'<div class="cp-card-title">{title}</div>', unsafe_allow_html=True)
            st.markdown(f'<p class="cp-card-text">{text}</p><span class="cp-pill {"demo" if tag == "Demonstration" else "ok"}">{tag}</span>', unsafe_allow_html=True)
            st.caption(f"Open the **{page}** tab above")

    st.markdown("### Reported results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Weather, 6 h temperature", "1.21 K", "vs 2.46 K persistence", delta_color="off", delta_arrow="off")
    m2.metric("El Niño TFT (5 seeds)", "0.78 °C", "vs 0.81 °C LSTM", delta_color="off", delta_arrow="off")
    m3.metric("PM2.5 RMSE", "46.6 µg/m³", "vs 52.0 persistence", delta_color="off", delta_arrow="off")
    m4.metric("Warming break", "1976", "0.035 → 0.211 °C/decade", delta_color="off", delta_arrow="off")
    ui.note(
        "Honest reading: the weather model halves persistence error but gains nothing from El Niño input; the TFT and LSTM are "
        "comparable; the pollution model shows no exceedance skill over always-alert on only 85 test readings."
    )

    st.markdown("### Checkpoints in use")
    rows = []
    for name, rel in MODELS:
        try:
            _, source = L.find_checkpoint(rel)
            rows.append({"Model": name, "Status": "found", "Source": source})
        except L.DataMissing:
            rows.append({"Model": name, "Status": "missing", "Source": "train it (make train) or restore a saved run"})
    st.dataframe(rows, hide_index=True, width="stretch")


TABS = [
    (":material/home: Overview", overview),
    (":material/public: Weather", weather.render),
    (":material/waves: El Niño", elnino.render),
    (":material/air: Air pollution", pollution.render),
    (":material/trending_up: Warming rate", warming.render),
]


def main() -> None:
    st.set_page_config(page_title="ClimatePulse", page_icon="🌍", layout="wide")
    ui.inject_css()
    tabs = st.tabs([label for label, _ in TABS], key="main_tabs", on_change="rerun")
    for tab, (_, render) in zip(tabs, TABS, strict=True):
        with tab:
            # Only the open tab runs, so one page's models are not loaded or re-run for the others
            if getattr(tab, "open", True):
                render()


main()
