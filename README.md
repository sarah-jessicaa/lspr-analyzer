# LSPR Spectrum Analyzer

**An automated feature-extraction and machine-learning pipeline for gelatin
authentication using Localized Surface Plasmon Resonance (LSPR) spectroscopy.**

Live demo: [ISI DENGAN URL STREAMLIT CLOUD SETELAH DEPLOYMENT]
Source: https://github.com/sarah-jessicaa/lspr-analyzer

---

## The problem

Gelatin is used in food, pharmaceuticals, and cosmetics, and its animal
origin (bovine, porcine, fish) matters for halal certification, religious
compliance, and allergen safety. Reference laboratory methods such as PCR,
HPLC, and FTIR can distinguish these origins, but they are slow, expensive,
and destructive to the sample.

LSPR spectroscopy is a rapid, low-cost, non-destructive alternative: each
gelatin sample produces an optical spectrum whose shape depends on the
sample's composition. The difficulty is that these spectra have broad, flat
resonance bands, so reading them by hand (locating peaks, measuring widths,
comparing curves) is repetitive, subjective, and error-prone.

This tool removes the manual step. Upload raw spectra; the application
stabilizes, quantifies, compares, and classifies them automatically.

## What this tool does

- **Stable peak detection.** Applies heavy Savitzky-Golay smoothing combined
  with a half-maximum band centroid, so peak positions remain stable even on
  flat, noisy plateaus. On our 72-spectrum dataset this reduced replicate
  peak variance from up to 77 nm to under 5 nm.
- **Automated feature extraction.** Computes 16 physically meaningful
  features per spectrum (see table below).
- **Species classification.** Trains and evaluates LDA, Random Forest, and
  XGBoost classifiers with stratified k-fold cross-validation, and reports
  confusion matrices, feature importance, and PCA/LDA projections.
- **Diagnostics.** Correlation heatmap between features and IsolationForest
  anomaly flags to catch measurement artifacts or mislabeled samples.
- **Comparison views.** Overlays spectra colored by species or concentration,
  with concentration filtering and per-species feature summaries.
- **ML-ready export.** Downloads a CSV of all features and metadata for any
  downstream workflow.

## How to use the web app

1. Open the live demo (link above) or run locally (see below).
2. Upload one or more spectrum files. Each file must contain two
   whitespace-separated columns: wavelength (nm) and intensity (a.u.).
   If you only want to try the interface, click **Download example file**
   on the start screen to get a correctly formatted dummy file.
3. Name files using the convention
   `<species><concentration>%-<replicate>.txt`, for example `sapi1%-1.txt`
   (bovine gelatin, 1% concentration, replicate 1). Recognized species
   labels: `sapi` (bovine), `babi` (porcine), `ikan` (fish). Files that do
   not match are skipped with an explanatory warning.
4. Adjust preprocessing in the sidebar (smoothing window, peak prominence
   ratio). The spectrum view and shape-based features update immediately;
   peak band center and FWHM intentionally use a fixed heavy smoothing so
   they stay stable.
5. Explore the four tabs:
   - **Spectrum** - raw signal (thin line) and smoothed signal (thick line)
     with the detected peak band marked.
   - **Features** - metric cards, a per-feature table with physical
     descriptions, and CSV export.
   - **Comparison** - overlay spectra by species or concentration, filter
     concentrations, or view mean feature vs concentration per species.
   - **Classification** - PCA/LDA projections, cross-validated accuracy for
     three models, confusion matrices, feature importance, correlation
     heatmap, and anomaly flags.
6. Download the CSV to feed any downstream machine-learning workflow.

## How to run locally

Requirements: Python 3.10 or newer.

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    .venv\Scripts\python.exe -m streamlit run app.py

On Linux/macOS use `.venv/bin/python` and `.venv/bin/streamlit` instead.
The app opens at http://localhost:8501.

## Extracted features

| Feature | Unit | Meaning |
| --- | --- | --- |
| Peak band center | nm | Wavelength at the center of the LSPR resonance band (half-max centroid) |
| Peak intensity | a.u. | Maximum intensity of the smoothed resonance |
| FWHM | nm | Width of the resonance band at half its maximum height |
| Q-factor | - | Peak band center / FWHM; higher means a sharper resonance |
| AUC | a.u.nm | Area under the smoothed curve over the measured range |
| Asymmetry | - | 0 = symmetric; positive = right tail; negative = left tail |
| Intensity at 450 / 550 / 650 nm | a.u. | Point intensities at fixed wavelengths |
| Regional slope | a.u./nm | Linear gradient over 400-500, 500-600, and 600-700 nm |
| Ratio 450/650 and 550/650 | - | Intensity ratios; shape descriptors largely free of concentration scaling |
| Centroid | nm | Intensity-weighted center of mass of the full spectrum |
| Skewness | - | Asymmetry of the spectral distribution around the centroid |

## Machine learning protocol

- **Dataset:** 72 spectra (3 species x 8 concentrations x 3 replicates),
  approximately 2955 wavelength points each, range roughly 380-1050 nm.
- **Features:** the 16 features above, standardized before modeling.
- **Models:** LDA, Random Forest (200 trees), XGBoost (200 boosting rounds).
- **Evaluation:** stratified 5-fold cross-validation reported as mean +/-
  standard deviation of accuracy. Confusion matrices shown in the app come
  from a single seeded 75/25 split and are illustrative, not the primary
  metric.
- **Honesty note:** with 72 samples, accuracy estimates carry real
  uncertainty. The app reports standard deviations and flags anomalous
  samples rather than presenting a single optimistic number.

## Why it exists

This project grew out of my internship research at BRIN (National Research
and Innovation Agency of Indonesia), where gelatin authentication relied on
manual spectral reading. The goal is practical: automate the repetitive
quantification so researchers can spend their time on interpretation.

The use of machine learning on broad-FWHM plasmonic spectra follows the
finding in Guo et al. (2024) that learned models can extract resonance
information from low-resolution, wide-band spectra more reliably than
manual peak picking; here that approach is adapted from regression on
fiber-SPR spectra to classification of LSPR gelatin spectra.

## Data note

Raw spectra are research data and are **not** included in this repository.
The application accepts any two-column whitespace-separated spectrum file;
the naming convention above enables automatic species, concentration, and
replicate labeling.

## Project structure

    lspr-analyzer/
    ├── app.py                  # Streamlit entry point and layout
    ├── config.py               # Central color palette
    ├── requirements.txt
    ├── .streamlit/config.toml  # Light theme configuration
    ├── assets/style.css        # Typography and component styling
    ├── core/
    │   ├── data_loader.py      # Parsing, labeling, caching
    │   ├── feature_extraction.py
    │   └── ml.py               # PCA/LDA/RF/XGBoost, CV, diagnostics
    └── ui/
        ├── theme.py            # Header and CSS injection
        ├── spectrum_viewer.py  # Spectrum tab
        ├── feature_results.py  # Features tab
        ├── comparison.py       # Comparison tab
        └── classification.py   # Classification tab

## Tech stack

Python, Streamlit, SciPy, NumPy, Pandas, scikit-learn, XGBoost, Plotly.

## Roadmap

- Excel wide-format input (one sheet, many spectra).
- Concentration regression (SVR or gradient boosting regressors).
- Recursive feature elimination (RFE) for a minimal feature subset.
- Explicit clustering analysis of unlabeled samples.

## License

Code: MIT License - Copyright (c) 2026 Sarah Jessica.
Permission is granted to use, copy, modify, and distribute this software
for any purpose, with attribution.

Data: no experimental data is distributed with this repository. Raw spectra
remain research data governed by the originating internship agreement and
are not licensed here.

## Author

Sarah Jessica 
