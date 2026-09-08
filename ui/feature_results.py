import numpy as np
import pandas as pd
import streamlit as st

FEATURE_INFO = {
    "peak_wavelength": ("Peak band center", "nm", "Wavelength at the center of the LSPR resonance band"),
    "peak_intensity": ("Peak intensity", "a.u.", "Maximum intensity of the smoothed resonance"),
    "fwhm": ("FWHM", "nm", "Full width at half maximum of the resonance band"),
    "q_factor": ("Q-factor", "", "Peak band center divided by FWHM; higher means a sharper resonance"),
    "auc": ("AUC", "a.u.nm", "Area under the smoothed curve over the measured range"),
    "asymmetry": ("Asymmetry", "", "0 = symmetric; positive = right tail; negative = left tail"),
    "intensity_450": ("Intensity @450nm", "a.u.", "Intensity at 450 nm"),
    "intensity_550": ("Intensity @550nm", "a.u.", "Intensity at 550 nm"),
    "intensity_650": ("Intensity @650nm", "a.u.", "Intensity at 650 nm"),
    "slope_400_500": ("Slope 400-500nm", "a.u./nm", "Linear gradient over 400-500 nm"),
    "slope_500_600": ("Slope 500-600nm", "a.u./nm", "Linear gradient over 500-600 nm"),
    "slope_600_700": ("Slope 600-700nm", "a.u./nm", "Linear gradient over 600-700 nm"),
    "ratio_450_650": ("Ratio 450/650", "", "Intensity ratio 450 to 650 nm; a shape descriptor largely free of concentration scaling"),
    "ratio_550_650": ("Ratio 550/650", "", "Intensity ratio 550 to 650 nm; a shape descriptor largely free of concentration scaling"),
    "centroid_wavelength": ("Centroid (nm)", "", "Intensity-weighted center of mass of the full spectrum"),
    "skewness": ("Skewness", "", "Asymmetry of the spectral distribution around the centroid"),
}

CARD_SPEC = [
    ("peak_wavelength", "Peak (nm)", lambda v: v),
    ("peak_intensity", "Intensity (a.u.)", lambda v: v),
    ("fwhm", "FWHM (nm)", lambda v: v),
    ("q_factor", "Q-factor", lambda v: v),
    ("auc", "AUC (×10³ a.u.nm)", lambda v: v / 1000.0),
]

FEATURE_GROUPS = [
    ("Resonance shape", ["peak_wavelength", "peak_intensity", "fwhm", "q_factor", "asymmetry"]),
    ("Signal integral", ["auc"]),
    ("Fixed-point intensities", ["intensity_450", "intensity_550", "intensity_650"]),
    ("Regional slopes", ["slope_400_500", "slope_500_600", "slope_600_700"]),
    ("Shape ratios", ["ratio_450_650", "ratio_550_650"]),
    ("Distribution moments", ["centroid_wavelength", "skewness"]),
]

def _zebra_style(df):
    return df.style.apply(
        lambda row: [
            "background-color: #f8fafc" if row.name % 2 else "background-color: #ffffff"
            for _ in row
        ],
        axis=1,
    )

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
        f"Cards and table below refer to the selected file: {filename}. "
        "The CSV download covers all uploaded files."
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
    st.caption("Grouped by physical category. The Description column explains the meaning of each value.")

    for group_title, keys in FEATURE_GROUPS:
        st.markdown(f"**{group_title}**")
        rows = []
        for key in keys:
            label, unit, desc = FEATURE_INFO[key]
            rows.append({
                "Feature": label,
                "Value": _fmt(features.get(key, np.nan)),
                "Unit": unit,
                "Description": desc,
            })
        st.dataframe(
            _zebra_style(pd.DataFrame(rows)),
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        "Download CSV (all files)",
        data=pd.DataFrame(all_rows).to_csv(index=False),
        file_name="lspr_features.csv",
        mime="text/csv",
    )