from pathlib import Path

import streamlit as st

CSS_PATH = Path(__file__).resolve().parent.parent / "assets" / "style.css"


def set_theme():
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


def render_header():
    st.markdown('<h1 class="header-title">LSPR Spectrum Analyzer</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="header-sub">Automated feature extraction pipeline for gelatin authentication</p>',
        unsafe_allow_html=True,
    )