import numpy as np
import pandas as pd
import streamlit as st

FEATURE_INFO = {
    "peak_wavelength": ("Peak band center", "nm", "Panjang gelombang pusat pita resonansi LSPR"),
    "peak_intensity": ("Peak intensity", "a.u.", "Intensitas maksimum pada puncak resonansi"),
    "fwhm": ("FWHM", "nm", "Full Width at Half Maximum, lebar pita pada setengah tinggi puncak"),
    "q_factor": ("Q-factor", "", "Rasio λmax terhadap FWHM, semakin tinggi semakin tajam resonansi"),
    "auc": ("AUC", "a.u.nm", "Area under curve, total energi spektrum di seluruh rentang"),
    "asymmetry": ("Asymmetry", "", "0 = simetris, positif = ekor ke kanan, negatif = ekor ke kiri"),
    "intensity_450": ("Intensity @450nm", "a.u.", "Intensitas pada panjang gelombang 450 nm"),
    "intensity_550": ("Intensity @550nm", "a.u.", "Intensitas pada panjang gelombang 550 nm"),
    "intensity_650": ("Intensity @650nm", "a.u.", "Intensitas pada panjang gelombang 650 nm"),
    "slope_400_500": ("Slope 400-500nm", "a.u./nm", "Gradien spektrum pada rentang 400-500 nm"),
    "slope_500_600": ("Slope 500-600nm", "a.u./nm", "Gradien spektrum pada rentang 500-600 nm"),
    "slope_600_700": ("Slope 600-700nm", "a.u./nm", "Gradien spektrum pada rentang 600-700 nm"),
    "ratio_450_650": ("Ratio 450/650", "", "Rasio intensitas 450 terhadap 650 nm, descriptor bentuk yang relatif bebas konsentrasi"),
    "ratio_550_650": ("Ratio 550/650", "", "Rasio intensitas 550 terhadap 650 nm, descriptor bentuk yang relatif bebas konsentrasi"),
    "centroid_wavelength": ("Centroid (nm)", "", "Pusat massa seluruh spektrum, tertimbang intensitas"),
    "skewness": ("Skewness", "", "Kemencengan distribusi spektral terhadap centroid"),
}

CARD_SPEC = [
    ("peak_wavelength", "Peak (nm)", lambda v: v),
    ("peak_intensity", "Intensity (a.u.)", lambda v: v),
    ("fwhm", "FWHM (nm)", lambda v: v),
    ("q_factor", "Q-factor", lambda v: v),
    ("auc", "AUC (×10³ a.u.nm)", lambda v: v / 1000.0),
]


def _fmt(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    if abs(value) < 1:
        return f"{value:.4f}"
    if abs(value) < 1000:
        return f"{value:,.2f}"
    return f"{value:,.1f}"


def show_feature_results(filename, features, all_rows):
    st.markdown('<div class="section-title">Core features</div>', unsafe_allow_html=True)
    st.caption(
        f"Kartu dan tabel di bawah merujuk pada file terpilih: {filename}. "
        "Unduhan CSV mencakup seluruh file yang di-upload."
    )

    cols = st.columns(len(CARD_SPEC), gap="small")
    for col, (key, title, scale) in zip(cols, CARD_SPEC):
        raw = features.get(key, np.nan)
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            value = np.nan
        else:
            value = scale(raw)
        with col:
            st.markdown(
                f'<div class="metric-card" style="min-height:118px">'
                f'<div class="metric-value">{_fmt(value)}</div>'
                f'<div class="metric-label">{title}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">All extracted features</div>', unsafe_allow_html=True)
    st.caption("Setiap baris adalah satu fitur. Kolom Description menjelaskan makna fisik dari angka tersebut.")

    rows = []
    for key, (label, unit, desc) in FEATURE_INFO.items():
        rows.append({
            "Feature": label,
            "Value": _fmt(features.get(key, np.nan)),
            "Unit": unit,
            "Description": desc,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.download_button(
        "Download CSV (all files)",
        data=pd.DataFrame(all_rows).to_csv(index=False),
        file_name="lspr_features.csv",
        mime="text/csv",
    )