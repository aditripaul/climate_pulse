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
        "Choose a simulation below or from the menu at the top.</p></div>",
        unsafe_allow_html=True,
    )
    cards = [
        (weather_page, "Weather model", ":material/public:", "Global 2 m temperature and rain, 6 hours ahead, animated against ERA5 on maps and a 3D globe.", "Validated for one step"),
        (elnino_page, "El Niño", ":material/waves:", "12-month Niño 3.4 forecasts with uncertainty bands from the TFT, compared with an LSTM.", "Validated"),
        (pollution_page, "Air pollution", ":material/air:", "Next-day PM2.5 at monitoring stations on a map, coloured by air-quality band.", "Demonstration"),
        (warming_page, "Warming rate", ":material/trending_up:", "Find when global warming changed speed, with adjustable break search.", "Statistical model"),
    ]
    cols = st.columns(4)
    for col, (page, title, icon, text, tag) in zip(cols, cards, strict=True):
        with col, st.container(border=True):
            st.markdown(f'<div class="cp-card-title">{title}</div>', unsafe_allow_html=True)
            st.markdown(f'<p class="cp-card-text">{text}</p><span class="cp-pill {"demo" if tag == "Demonstration" else "ok"}">{tag}</span>', unsafe_allow_html=True)
            st.page_link(page, label="Open", icon=icon)

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


overview_page = st.Page(overview, title="Overview", icon=":material/home:", url_path="overview", default=True)
weather_page = st.Page(weather.render, title="Weather", icon=":material/public:", url_path="weather")
elnino_page = st.Page(elnino.render, title="El Niño", icon=":material/waves:", url_path="elnino")
pollution_page = st.Page(pollution.render, title="Air pollution", icon=":material/air:", url_path="pollution")
warming_page = st.Page(warming.render, title="Warming rate", icon=":material/trending_up:", url_path="warming")


def main() -> None:
    st.set_page_config(page_title="ClimatePulse", page_icon="🌍", layout="wide")
    ui.inject_css()
    st.navigation([overview_page, weather_page, elnino_page, pollution_page, warming_page], position="top").run()


main()
