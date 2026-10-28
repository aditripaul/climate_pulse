"""Shared page chrome: CSS, headers and small layout helpers (white, grey, blue)."""

from __future__ import annotations

import streamlit as st

CSS = """
<style>
.block-container {padding-top: 4.6rem; padding-bottom: 3rem; max-width: 1280px;}
h1, h2, h3 {letter-spacing: -0.01em; color: #1F2937;}
/* Metric cards */
[data-testid="stMetric"] {background: #F3F4F6; border: 1px solid #E5E7EB; border-radius: 14px; padding: 14px 18px;}
[data-testid="stMetricLabel"] p {color: #6B7280; font-size: 0.85rem;}
[data-testid="stMetricValue"] {color: #1F2937; font-weight: 600;}
/* Bordered containers act as control panels */
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 14px; border-color: #E5E7EB;}
/* Top navigation */
header[data-testid="stHeader"] {background: rgba(255,255,255,0.92); border-bottom: 1px solid #E5E7EB;}
/* Hero + pills */
.cp-hero {background: linear-gradient(135deg, #F3F6FB 0%, #FFFFFF 70%); border: 1px solid #E5E7EB; border-radius: 18px; padding: 2rem 2.2rem; margin-bottom: 1.2rem;}
.cp-hero h1 {margin: 0 0 .3rem 0; font-size: 2.4rem;}
.cp-hero p {color: #4B5563; font-size: 1.05rem; margin: 0; max-width: 52rem;}
.cp-eyebrow {color: #2F6DB5; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; font-size: .78rem;}
.cp-sub {color: #4B5563; font-size: 1.02rem; margin: -.4rem 0 1rem 0; max-width: 52rem;}
.cp-pill {display: inline-block; padding: 2px 12px; border-radius: 999px; font-size: .8rem; font-weight: 600; margin-left: .6rem; vertical-align: middle;}
.cp-pill.ok {background: #E3ECF8; color: #1E4E8C;}
.cp-pill.demo {background: #ECEEF1; color: #4B5563;}
.cp-card-title {font-weight: 600; font-size: 1.1rem; color: #1F2937;}
.cp-card-text {color: #4B5563; font-size: .95rem;}
.cp-note {border-left: 4px solid #2F6DB5; background: #F3F6FB; padding: .7rem 1rem; border-radius: 6px; color: #374151; font-size: .95rem;}
.cp-foot {color: #6B7280; font-size: .82rem;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(eyebrow: str, title: str, subtitle: str, badge: str | None = None, badge_kind: str = "ok") -> None:
    pill = f'<span class="cp-pill {badge_kind}">{badge}</span>' if badge else ""
    st.markdown(f'<div class="cp-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f"## {title} {pill}", unsafe_allow_html=True)
    st.markdown(f'<p class="cp-sub">{subtitle}</p>', unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="cp-note">{text}</div>', unsafe_allow_html=True)


def footnote(text: str) -> None:
    st.markdown(f'<p class="cp-foot">{text}</p>', unsafe_allow_html=True)


def choice(label: str, options: list[str], default: str, key: str, help: str | None = None) -> str:
    """Segmented single choice that never returns None."""
    value = st.segmented_control(label, options, default=default, key=key, help=help)
    return value or default
