from pathlib import Path

import streamlit as st

from config import ACCENT, GRID, INK, LINE, MUTED

CSS_PATH = Path(__file__).resolve().parent.parent / "assets" / "style.css"

ROOT_VARS = f"""
:root {{
    --ink: {INK};
    --muted: {MUTED};
    --line: {LINE};
    --grid: {GRID};
    --accent: {ACCENT};
}}
"""


def set_theme():
    st.markdown(f"<style>{ROOT_VARS}</style>", unsafe_allow_html=True)
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


def render_header():
    st.markdown('<h1 class="header-title">LSPR Spectrum Analyzer</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="header-sub">Automated feature extraction pipeline for gelatin authentication</p>',
        unsafe_allow_html=True,
    )