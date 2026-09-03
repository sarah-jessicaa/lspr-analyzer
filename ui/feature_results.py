import numpy as np
import pandas as pd
import streamlit as st

FEATURE_INFO = {
    "peak_wavelength": ("Peak band center", "nm", "Panjang gelombang pusat pita resonansi LSPR"),
    "peak_intensity": ("Peak intensity", "a.u.", "Intensitas maksimum pada puncak resonansi"),
    "fwhm": ("FWHM", "nm", "Full Width at Half Maximum — lebar pita pada setengah tinggi puncak"),
    "q_factor": ("Q-factor", "", "Rasio λmax/FWHM — semakin tinggi, resonansi semakin tajam"),
    "auc": ("AUC", "a.u.·nm", "Area under curve — total energi spektrum di seluruh rentang"),
    "asymmetry": ("Asymmetry", "", "0 = simetris; positif = ekor ke kanan; negatif = ekor ke kiri"),
    "intensity_450": ("Intensity @450nm", "a.u.", "Intensitas pada panjang gelombang 450 nm"),
    "intensity_550": ("Intensity @550nm", "a.u.", "Intensitas pada panjang gelombang 550 nm"),
    "intensity_650": ("Intensity @650nm", "a.u.", "Intensitas pada panjang gelombang 650 nm"),
    "slope_400_500": ("Slope 400-500nm", "a.u./nm", "Gradien spektrum pada rentang 400-500 nm"),
    "slope_500_600": ("Slope 500-600nm", "a.u./nm", "Gradien spektrum pada rentang 500-600 nm"),
    "slope_600_700": ("Slope 600-700nm", "a.u./nm", "Gradien spektrum pada rentang 600-700 nm"),
}


def _fmt(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if abs(value) < 0.01 and value != 0:
        return f"{value:.4f}"
    if abs(value) < 1:
        return f"{value:.4f}"
    if abs(value) < 1000:
        return f"{value:,.2f}"
    return f"{value:,.1f}"


def show_feature_results(features, all_rows):
    st.markdown('<div class="section-title">Core features</div>', unsafe_allow_html=True)

    cards = [
        ("Peak band center", "peak_wavelength", "nm"),
        ("Peak intensity", "peak_intensity", "a.u."),
        ("FWHM", "fwhm", "nm"),
        ("Q-factor", "q_factor", ""),
        ("AUC", "auc", "a.u.nm"),
    ]
    cols = st.columns(len(cards))
    for col, (label, key, unit) in zip(cols, cards):
        value = features.get(key, np.nan)
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{_fmt(value)}</div>'
                f'<div class="metric-label">{label} {unit}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">All extracted features</div>', unsafe_allow_html=True)
    st.caption("Setiap baris adalah satu fitur. Kolom Description menjelaskan makna fisik dari angka tersebut.")

    rows = []
    for key in FEATURE_INFO:
        label, unit, desc = FEATURE_INFO[key]
        rows.append({
            "Feature": label,
            "Value": _fmt(features.get(key, np.nan)),
            "Unit": unit,
            "Description": desc,
        })
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Feature": st.column_config.TextColumn("Feature", width="medium"),
            "Value": st.column_config.TextColumn("Value", width="small"),
            "Unit": st.column_config.TextColumn("Unit", width="small"),
            "Description": st.column_config.TextColumn("Description", width="large"),
        },
    )

    st.download_button(
        "Download CSV (all files)",
        data=pd.DataFrame(all_rows).to_csv(index=False),
        file_name="lspr_features.csv",
        mime="text/csv",
    )