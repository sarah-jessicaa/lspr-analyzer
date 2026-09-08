from collections import Counter

import plotly.graph_objects as go
import streamlit as st

from config import ACCENT, INK, SPECIES_COLORS
from core.ml import FEATURE_KEYS, analyze
from ui.spectrum_viewer import base_layout


def show_classification(all_rows):
    counts = Counter(row["species"] for row in all_rows)
    if len(counts) < 2 or min(counts.values()) < 2:
        st.info("Classification needs at least two species with at least two spectra each.")
        return

    r = analyze(all_rows)

    st.caption(
        f"{r['n_total']} spectra. Primary metric: stratified {r['folds']}-fold CV "
        f"(mean ± std). Confusion matrices are from a single {r['n_train']}/{r['n_test']} "
        "seeded split, illustrative for this sample size."
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
        "PCA uses no labels; LDA uses labels to maximize separation between species. "
        "Hover a point to see its file name. Both are descriptive views; accuracy "
        "numbers come from cross-validation."
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
    n_models = len(names)

    if n_models == 1:
        _, mid, _ = st.columns([1, 2, 1], gap="large")
        with mid:
            st.plotly_chart(
                _confusion_fig(r["models"][names[0]]["confusion"], r["labels"], names[0]),
                use_container_width=True, theme=None,
            )
    elif n_models == 2:
        cols = st.columns(2, gap="large")
        for i, name in enumerate(names):
            with cols[i]:
                st.plotly_chart(
                    _confusion_fig(r["models"][name]["confusion"], r["labels"], name),
                    use_container_width=True, theme=None,
                )
    else:
        # 3 or more models: 2 on top, rest dynamically below
        cols_top = st.columns(2, gap="large")
        for i in range(2):
            with cols_top[i]:
                st.plotly_chart(
                    _confusion_fig(r["models"][names[i]]["confusion"], r["labels"], names[i]),
                    use_container_width=True, theme=None,
                )
        
        remaining = names[2:]
        if len(remaining) == 1:
            _, mid, _ = st.columns([1, 2, 1], gap="large")
            with mid:
                st.plotly_chart(
                    _confusion_fig(r["models"][remaining[0]]["confusion"], r["labels"], remaining[0]),
                    use_container_width=True, theme=None,
                )
        elif len(remaining) > 1:
            cols_bottom = st.columns(len(remaining), gap="large")
            for i, name in enumerate(remaining):
                with cols_bottom[i]:
                    st.plotly_chart(
                        _confusion_fig(r["models"][name]["confusion"], r["labels"], name),
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
            "X axis: relative contribution of each feature to the model decision; "
            "all features sum to 1. Y axis: feature name."
        )

    st.markdown('<div class="section-title">Feature selection diagnostics</div>', unsafe_allow_html=True)
    col_corr, col_anom = st.columns([3, 2], gap="large")
    with col_corr:
        st.plotly_chart(_corr_fig(r["corr"]), use_container_width=True, theme=None)
        st.caption(
            "Blue: strong positive correlation, red: negative. Pairs near 1 are redundant. "
            "Variance threshold and RFE are deferred post-deadline as these 16 features "
            "are already curated via domain knowledge."
        )
    with col_anom:
        st.markdown("**Anomaly detection (IsolationForest)**")
        if r["anomalies"]:
            for fn, sc in r["anomalies"]:
                st.markdown(f"- `{fn}` (score {sc:.3f})")
            st.caption(
                "Flagged samples deserve a manual check: possible measurement artifact "
                "or mislabeling, not necessarily a model failure."
            )
        else:
            st.caption("No samples flagged as anomalies.")


def _scatter(coords, xtitle, ytitle, r):
    fig = go.Figure()
    for sp in r["labels"]:
        mask = r["species"] == sp
        fig.add_trace(go.Scatter(
            x=coords[mask, 0], y=coords[mask, 1],
            mode="markers", name=sp,
            marker=dict(color=SPECIES_COLORS.get(sp, INK),
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
        title=dict(text=title, font=dict(size=14, color=INK)),
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
        marker=dict(color=ACCENT),
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


def _corr_fig(corr):
    fig = go.Figure(go.Heatmap(
        z=corr, x=FEATURE_KEYS, y=FEATURE_KEYS,
        colorscale="RdBu", zmin=-1, zmax=1,
        xgap=1, ygap=1,
    ))
    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=20, b=90),
        xaxis=dict(tickfont=dict(size=8), tickangle=-55),
        yaxis=dict(tickfont=dict(size=8), autorange="reversed"),
    )
    return fig