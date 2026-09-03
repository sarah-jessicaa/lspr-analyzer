import numpy as np
import plotly.graph_objects as go
import streamlit as st

SPECIES_COLORS = {"babi": "#c53030", "ikan": "#2b6cb0", "sapi": "#2f855a"}


def base_layout(fig, show_legend=False, hovermode="closest"):
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Segoe UI, Helvetica Neue, Arial", color="#1a365d"),
        xaxis=dict(gridcolor="#e8edf4", title="Wavelength (nm)"),
        yaxis=dict(gridcolor="#e8edf4", title="Intensity (a.u.)"),
        height=420,
        margin=dict(l=50, r=20, t=30, b=50),
        hovermode=hovermode,
        showlegend=show_legend,
    )
    return fig


def show_spectrum_viewer(df, label, features, y_smooth):
    color = SPECIES_COLORS.get(label["species"], "#1a365d")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.wavelength, y=df.intensity,
        mode="lines", name="raw",
        line=dict(color=color, width=0.8), opacity=0.35,
        hovertemplate="raw<br>%{x:.1f} nm, %{y:.1f} a.u.<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df.wavelength, y=y_smooth,
        mode="lines", name="smoothed",
        line=dict(color=color, width=2),
        hovertemplate="smoothed<br>%{x:.1f} nm, %{y:.1f} a.u.<extra></extra>",
    ))

    peak_wl = features.get("peak_wavelength")
    if peak_wl is not None and not np.isnan(peak_wl):
        fig.add_vline(
            x=peak_wl, line_dash="dash", line_color="#64748b",
            annotation_text="peak band center",
        )

    base_layout(fig, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, theme=None)