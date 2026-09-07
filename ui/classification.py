from collections import Counter

import plotly.graph_objects as go
import streamlit as st

from core.ml import analyze
from ui.spectrum_viewer import SPECIES_COLORS, base_layout


def show_classification(all_rows):
    counts = Counter(row["species"] for row in all_rows)
    if len(counts) < 2 or min(counts.values()) < 2:
        st.info("Klasifikasi butuh minimal 2 spesies dengan minimal 2 spektrum masing-masing.")
        return

    r = analyze(all_rows)

    st.caption(
        f"{r['n_total']} spektrum. Angka utama: stratified {r['folds']}-fold CV "
        f"(mean ± std). Matriks konfusi dari satu split {r['n_train']}/{r['n_test']} "
        "ber-seed tetap, bersifat ilustratif untuk ukuran sampel ini."
    )

    st.markdown('<div class="section-title">Projections</div>', unsafe_allow_html=True)
    st.plotly_chart(
        _scatter(r["pca_coords"],
                 f"PC1 ({r['explained'][0]:.0%})",
                 f"PC2 ({r['explained'][1]:.0%})", r),
        use_container_width=True, theme=None,
    )
    st.plotly_chart(
        _scatter(r["lda_coords"], "LD1", "LD2", r),
        use_container_width=True, theme=None,
    )
    st.caption(
        "PCA tidak memakai label; LDA memakai label untuk memaksimalkan pemisahan "
        "antar spesies. Arahkan kursor ke titik untuk melihat nama file. Keduanya "
        "visualisasi deskriptif; angka akurasi tetap berasal dari cross-validation."
    )

    st.markdown('<div class="section-title">Classification accuracy (CV)</div>', unsafe_allow_html=True)
    names = list(r["models"].keys())
    cols = st.columns(len(names), gap="large")
    for col, name in zip(cols, names):
        m = r["models"][name]
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">'
                f'{m["cv_mean"]:.0%} ± {m["cv_std"]:.0%}</div>'
                f'<div class="metric-label">{name}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Confusion matrices (illustrative split)</div>', unsafe_allow_html=True)
    cols = st.columns(2, gap="large")
    with cols[0]:
        st.plotly_chart(
            _confusion_fig(r["models"][names[0]]["confusion"], r["labels"], names[0]),
            use_container_width=True, theme=None,
        )
    with cols[1]:
        st.plotly_chart(
            _confusion_fig(r["models"][names[1]]["confusion"], r["labels"], names[1]),
            use_container_width=True, theme=None,
        )
    _, mid, _ = st.columns([1, 2, 1], gap="large")
    with mid:
        st.plotly_chart(
            _confusion_fig(r["models"][names[2]]["confusion"], r["labels"], names[2]),
            use_container_width=True, theme=None,
        )

    if r["importance"]:
        st.markdown('<div class="section-title">Feature importance</div>', unsafe_allow_html=True)
        source = st.selectbox("Source", list(r["importance"].keys()))
        st.plotly_chart(
            _importance_fig(r["importance"][source]),
            use_container_width=True, theme=None,
        )
        st.caption(
            "Sumbu x: kontribusi relatif fitur terhadap keputusan model; "
            "seluruh fitur dijumlahkan sama dengan 1. Sumbu y: nama fitur."
        )


def _scatter(coords, xtitle, ytitle, r):
    fig = go.Figure()
    for sp in r["labels"]:
        mask = r["species"] == sp
        fig.add_trace(go.Scatter(
            x=coords[mask, 0], y=coords[mask, 1],
            mode="markers", name=sp,
            marker=dict(color=SPECIES_COLORS.get(sp, "#1a365d"),
                        size=9, opacity=0.8),
            customdata=r["filenames"][mask],
            hovertemplate="%{customdata}<br>%{x:.2f}, %{y:.2f}<extra></extra>",
        ))
    base_layout(fig, show_legend=True)
    fig.update_xaxes(title=xtitle)
    fig.update_yaxes(title=ytitle)
    return fig


def _confusion_fig(cm, labels, title, height=340):
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Blues", showscale=False,
        xgap=3, ygap=3, texttemplate="%{z}",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#1a365d")),
        height=height,
        margin=dict(l=70, r=10, t=50, b=60),
        xaxis=dict(title="Predicted", tickfont=dict(size=11)),
        yaxis=dict(title="Actual", autorange="reversed",
                   tickfont=dict(size=11)),
    )
    return fig


def _importance_fig(importance):
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color="#2563eb"),
    ))
    fig.update_layout(
        height=420,
        margin=dict(l=150, r=20, t=20, b=50),
        xaxis=dict(title="Relative importance (total = 1)"),
        yaxis=dict(categoryorder="total ascending",
                   tickfont=dict(size=11)),
        showlegend=False,
    )
    return fig