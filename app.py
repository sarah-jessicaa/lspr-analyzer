import streamlit as st

from core.data_loader import load_spectrum, parse_label
from core.feature_extraction import extract_features, smooth
from ui.classification import show_classification
from ui.comparison import show_comparison
from ui.feature_results import show_feature_results
from ui.spectrum_viewer import show_spectrum_viewer
from ui.theme import render_header, set_theme

st.set_page_config(page_title="LSPR Analyzer", layout="wide")
set_theme()
render_header()

with st.sidebar:
    st.markdown('<div class="section-title">Data input</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload spectrum files (.txt)",
        type=["txt", "csv"],
        accept_multiple_files=True,
        help="Two whitespace-separated columns: wavelength (nm) and intensity (a.u.).",
    )
    st.caption(
        "File name convention: species-concentration-replicate, "
        "for example sapi1%-1.txt (species labels: sapi, babi, ikan)."
    )

    st.markdown('<div class="section-title">Preprocessing</div>', unsafe_allow_html=True)
    window = st.slider("Smoothing window", 5, 51, 11, step=2)
    prominence_ratio = st.slider("Peak prominence ratio", 0.01, 0.20, 0.05, step=0.01)

    st.markdown('<div class="section-title">Comparison</div>', unsafe_allow_html=True)
    color_by = st.radio(
        "Color by",
        ["auto", "species", "concentration"],
        horizontal=True,
    )

spectra = {}
for file in uploaded or []:
    label = parse_label(file.name)
    if label is None:
        st.warning(f"Unrecognized file name: {file.name}")
        continue
    try:
        df = load_spectrum(file)
    except ValueError as e:
        st.warning(str(e))
        continue
    spectra[file.name] = {"label": label, "df": df}

if not spectra:
    st.markdown(
        "Upload one or more spectrum files to begin. Each file must contain "
        "two whitespace-separated columns: wavelength (nm) and intensity (a.u.)."
    )
    st.markdown(
        "File names encode the sample as species-concentration-replicate, "
        "for example sapi1%-1.txt for bovine gelatin at 1 percent, replicate 1. "
        "Recognized species labels: sapi, babi, ikan."
    )
    st.stop()

all_rows = []
features_by_file = {}
for name, item in spectra.items():
    df = item["df"]
    feats = extract_features(
        df.wavelength.to_numpy(),
        df.intensity.to_numpy(),
        window=window,
        prominence_ratio=prominence_ratio,
    )
    features_by_file[name] = feats
    all_rows.append({"filename": name, **item["label"], **feats})

tab_view, tab_feat, tab_comp, tab_cls = st.tabs(
    ["Spectrum", "Features", "Comparison", "Classification"]
)

with tab_view:
    options = list(spectra.keys())
    chosen = st.selectbox("File", options)
    if chosen is None:
        chosen = options[0]
    item = spectra[chosen]
    y_smooth = smooth(item["df"].intensity, window=window)
    show_spectrum_viewer(item["df"], item["label"], features_by_file[chosen], y_smooth)

with tab_feat:
    show_feature_results(chosen, features_by_file[chosen], all_rows)

with tab_comp:
    show_comparison(spectra, all_rows, color_by)

with tab_cls:
    show_classification(all_rows)