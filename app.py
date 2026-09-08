import numpy as np
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
        "File name convention: `[species][concentration]%-[replicate].txt` "
        "(e.g., `sapi1%-1.txt`). Valid species: sapi, babi, ikan."
    )

    st.markdown('<div class="section-title">Preprocessing</div>', unsafe_allow_html=True)
    window = st.slider("Smoothing window", 5, 51, 11, step=2)
    prominence_ratio = st.slider("Peak prominence ratio", 0.01, 0.20, 0.05, step=0.01)
    st.caption(
        "Peak band center and FWHM use a fixed heavy smoothing for stability; "
        "this window controls the displayed curve and shape-based features."
    )

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
        st.warning(
            f"File `{file.name}` skipped. Expected format: `[species][concentration]%-[replicate].txt` "
            f"(e.g., `sapi1%-1.txt`). Valid species: sapi, babi, ikan."
        )
        continue
    try:
        df = load_spectrum(file)
    except ValueError as e:
        st.warning(f"Error reading `{file.name}`: {e}")
        continue
    spectra[file.name] = {"label": label, "df": df}

if not spectra:
    st.markdown(
        "Upload one or more spectrum files to begin. Each file must contain "
        "two whitespace-separated columns: wavelength (nm) and intensity (a.u.)."
    )
    st.markdown(
        "**File naming convention:** `[species][concentration]%-[replicate].txt` "
        "(e.g., `sapi1%-1.txt`). Recognized species: `sapi`, `babi`, `ikan`."
    )
    
    # Dummy example file for UX guidance: synthetic peak, realistic length
    _wl = np.arange(400.0, 700.0, 2.0)
    _int = 200.0 + 150.0 * np.exp(-0.5 * ((_wl - 550.0) / 40.0) ** 2)
    example_data = "\n".join(f"{w:.1f} {i:.2f}" for w, i in zip(_wl, _int))
    st.download_button(
        label="Download example file (sapi1%-1.txt)",
        data=example_data,
        file_name="sapi1%-1.txt",
        mime="text/plain",
        help="A minimal example showing the required two-column format and naming convention.",
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

options = list(spectra.keys())

tab_view, tab_feat, tab_comp, tab_cls = st.tabs(
    ["Spectrum", "Features", "Comparison", "Classification"]
)

with tab_view:
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