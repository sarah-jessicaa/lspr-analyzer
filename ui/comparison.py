import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import CONC_SCALE, INK, SPECIES_COLORS
from ui.spectrum_viewer import base_layout


def resolve_color_mode(spectra, requested):
    if requested != "auto":
        return requested
    species = {item["label"]["species"] for item in spectra.values()}
    return "concentration" if len(species) == 1 else "species"


def show_comparison(spectra, all_rows, color_by):
    if len(spectra) < 2:
        st.info("Upload at least two files to compare spectra.")
        return

    mode = resolve_color_mode(spectra, color_by)
    view = st.radio("View", ["Spectra", "Feature summary"], horizontal=True)

    if view == "Spectra":
        concs = sorted({item["label"]["concentration"] for item in spectra.values()})
        chosen_concs = st.multiselect("Concentration filter", concs, default=concs)
        shown = {
            name: item for name, item in spectra.items()
            if item["label"]["concentration"] in chosen_concs
        }
        if len(shown) < 2:
            st.info("Select at least two concentrations to compare.")
            return

        seen_groups = set()
        fig = go.Figure()
        for name, item in shown.items():
            label = item["label"]
            if mode == "species":
                color = SPECIES_COLORS.get(label["species"], INK)
                group = label["species"]
                legend_name = label["species"]
            else:
                idx = min(max(label["concentration"] - 1, 0), 7)
                color = CONC_SCALE[idx]
                group = label["concentration"]
                legend_name = f"{label['concentration']}%"
            first = group not in seen_groups
            seen_groups.add(group)
            fig.add_trace(go.Scatter(
                x=item["df"].wavelength, y=item["df"].intensity,
                mode="lines", name=legend_name,
                line=dict(color=color, width=1), opacity=0.8,
                legendgroup=str(group), showlegend=first,
                hovertemplate=(
                    f"{label['species']} {label['concentration']}% "
                    f"rep {label['replicate']}<br>"
                    + "%{x:.1f} nm, %{y:.1f} a.u.<extra></extra>"
                ),
            ))
        hover = "x unified" if len(shown) <= 6 else "closest"
        base_layout(fig, show_legend=True, hovermode=hover)
        st.plotly_chart(fig, use_container_width=True, theme=None)
        return

    df = pd.DataFrame(all_rows)
    metric = st.selectbox("Metric", ["peak_intensity", "auc", "fwhm", "q_factor"])

    fig = go.Figure()
    for species, group in df.groupby("species"):
        agg = group.groupby("concentration")[metric].mean()
        fig.add_trace(go.Scatter(
            x=agg.index, y=agg.values,
            mode="lines+markers", name=species,
            line=dict(color=SPECIES_COLORS.get(species, INK), width=2),
        ))
    base_layout(fig, show_legend=True)
    fig.update_xaxes(title="Concentration (%)")
    fig.update_yaxes(title=f"Mean {metric}")
    st.plotly_chart(fig, use_container_width=True, theme=None)