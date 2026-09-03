# LSPR Spectrum Analyzer

Automated feature extraction pipeline for gelatin authentication from LSPR spectra.

## What it does

Upload raw LSPR spectra (two columns: wavelength and intensity, whitespace-separated) and the app will:

- extract physical features per spectrum: peak band center, peak intensity, FWHM, Q-factor, AUC, asymmetry, regional slopes, intensity at fixed wavelengths
- stabilize peak detection on flat plateaus via heavy Savitzky-Golay smoothing and a half-max band centroid
- compare spectra across species (sapi / babi / ikan) and concentrations with adaptive color encoding
- export an ML-ready CSV of all extracted features

## Why it exists

Grew out of my internship research at BRIN, where gelatin authentication relied on manual spectral reading. This tool automates the repetitive part so researchers can focus on interpretation.

## Tech stack

Python, Streamlit, SciPy, NumPy, Pandas, Plotly

## Run locally

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    .venv\Scripts\python.exe -m streamlit run app.py

## Data

Raw spectra are not included (research data). Any two-column spectrum file works; filenames like sapi1%-1.txt (species, concentration, replicate) enable automatic labeling.

## Roadmap

- Excel wide-format input
- Species classification (LDA / Random Forest) on extracted features
- Anomaly detection for outlier spectra

## Author

Sarah Jessica - Physics, Universitas Diponegoro