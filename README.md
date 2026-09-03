# LSPR Spectrum Analyzer

An automated feature-extraction pipeline for gelatin authentication using
Localized Surface Plasmon Resonance (LSPR) spectroscopy.

Live demo: [add your Streamlit Cloud URL after deployment]

## The problem

Gelatin from different animal sources (bovine, porcine, fish) must be
distinguished for halal certification and allergen safety. Reference
methods such as PCR, HPLC, and FTIR are accurate but slow, expensive,
and destructive. LSPR spectroscopy is a rapid, low-cost, non-destructive
alternative - but reading and comparing LSPR spectra manually is
repetitive and error-prone.

## What this tool does

- Extracts physically meaningful features from each spectrum: peak band
  center, peak intensity, full width at half maximum (FWHM), quality
  factor (Q-factor), area under the curve (AUC), peak asymmetry,
  regional slopes, and intensities at fixed wavelengths
- Stabilizes peak detection on flat, noisy plateaus using heavy
  Savitzky-Golay smoothing combined with a half-maximum band centroid
- Compares spectra across species (bovine / porcine / fish) and
  concentrations (1-8%) with adaptive color encoding
- Exports an ML-ready CSV of all extracted features for downstream
  classification

## How to use the web app

1. Upload one or more spectrum files (.txt with two whitespace-separated
   columns: wavelength and intensity).
2. File names encode the sample metadata as
   <species><concentration>%-<replicate>.txt, for example sapi1%-1.txt
   means bovine gelatin, 1% concentration, replicate 1. Recognized
   species labels: sapi (bovine), babi (porcine), ikan (fish).
3. Adjust preprocessing in the sidebar (smoothing window, peak
   prominence ratio). The spectrum view updates immediately.
4. Explore the three tabs:
   - Spectrum: raw signal (thin line) and smoothed signal (thick line)
     with the detected peak band marked.
   - Features: metric cards, a per-feature table with physical
     descriptions, and CSV export.
   - Comparison: overlay spectra colored by species or concentration,
     filter concentrations, or switch to a feature summary (mean
     feature vs concentration, per species).
5. Download the CSV to feed any downstream machine-learning workflow.

## How to run locally

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    .venv\Scripts\python.exe -m streamlit run app.py

(On Linux/macOS use .venv/bin/python and .venv/bin/streamlit instead.)

## Extracted features

| Feature | Unit | Meaning |
| --- | --- | --- |
| Peak band center | nm | Wavelength at the center of the LSPR resonance band |
| Peak intensity | a.u. | Maximum intensity of the resonance |
| FWHM | nm | Width of the resonance band at half its maximum height |
| Q-factor | - | Peak / FWHM; higher means a sharper resonance |
| AUC | a.u.nm | Total spectral energy over the measured range |
| Asymmetry | - | 0 = symmetric; positive = right tail; negative = left tail |
| Intensity at 450/550/650 nm | a.u. | Point intensities at fixed wavelengths |
| Regional slope | a.u./nm | Gradient over 400-500, 500-600, and 600-700 nm |

## Why it exists

Grew out of my internship research at BRIN (National Research and
Innovation Agency of Indonesia), where gelatin authentication relied on
manual spectral reading. This tool automates the repetitive part so
researchers can focus on interpretation.

## Data note

Raw spectra are research data and are not included in this repository.
The app accepts any two-column whitespace-separated spectrum file; the
naming convention above enables automatic labeling.

## Tech stack

Python, Streamlit, SciPy, NumPy, Pandas, Plotly

## Roadmap

- Excel wide-format input
- Species classification (LDA / Random Forest) on extracted features
- Anomaly detection for outlier spectra

## Author

Sarah Jessica 