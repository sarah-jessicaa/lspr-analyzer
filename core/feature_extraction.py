import numpy as np
from scipy.integrate import simpson
from scipy.signal import find_peaks, peak_widths, savgol_filter

FEATURE_COLUMNS = [
    "peak_wavelength", "peak_intensity", "fwhm", "auc",
    "q_factor", "asymmetry",
    "intensity_450", "intensity_550", "intensity_650",
    "slope_400_500", "slope_500_600", "slope_600_700",
    "ratio_450_650", "ratio_550_650", "centroid_wavelength", "skewness",
]


def smooth(intensity, window=11, polyorder=2):
    return savgol_filter(np.asarray(intensity, dtype=float), window, polyorder)


def extract_features(wavelength, intensity, window=11, polyorder=2, prominence_ratio=0.05):
    wavelength = np.asarray(wavelength, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    y = savgol_filter(intensity, window, polyorder)

    # smoothing tebal khusus deteksi peak, agar stabil di plateau datar
    det_window = 151 if intensity.size > 151 else max(5, ((intensity.size - 1) // 2) * 2 + 1)
    y_det = savgol_filter(intensity, det_window, 3)
    prominence = prominence_ratio * (y_det.max() - y_det.min())

    feats = {name: np.nan for name in FEATURE_COLUMNS}

    peaks, _ = find_peaks(y_det, prominence=prominence)
    if len(peaks) == 0:
        return feats

    main = peaks[np.argmax(y_det[peaks])]

    _, _, left_ips, right_ips = peak_widths(y_det, [main], rel_height=0.5)
    left_wl = np.interp(left_ips[0], np.arange(wavelength.size), wavelength)
    right_wl = np.interp(right_ips[0], np.arange(wavelength.size), wavelength)
    width = right_wl - left_wl

    mask = (wavelength >= left_wl) & (wavelength <= right_wl)
    weights = y_det[mask] - y_det[mask].min()
    peak_wl = np.average(wavelength[mask], weights=weights) if weights.sum() > 0 else wavelength[main]

    feats["peak_wavelength"] = peak_wl
    feats["peak_intensity"] = y[main]
    feats["fwhm"] = width
    feats["auc"] = simpson(y, x=wavelength)
    feats["q_factor"] = peak_wl / width if width > 0 else np.nan
    if width > 0:
        feats["asymmetry"] = (right_wl + left_wl - 2 * peak_wl) / width

    for target in (450, 550, 650):
        feats[f"intensity_{target}"] = np.interp(target, wavelength, y)

    for lo, hi in ((400, 500), (500, 600), (600, 700)):
        m = (wavelength >= lo) & (wavelength <= hi)
        if m.sum() > 1:
            feats[f"slope_{lo}_{hi}"] = np.polyfit(wavelength[m], y[m], 1)[0]

    i650 = feats["intensity_650"]
    if i650 is not None and not np.isnan(i650) and i650 != 0:
        feats["ratio_450_650"] = feats["intensity_450"] / i650
        feats["ratio_550_650"] = feats["intensity_550"] / i650

    w = y - y.min()
    total = w.sum()
    if total > 0:
        centroid = np.average(wavelength, weights=w)
        m2 = np.average((wavelength - centroid) ** 2, weights=w)
        m3 = np.average((wavelength - centroid) ** 3, weights=w)
        feats["centroid_wavelength"] = centroid
        feats["skewness"] = m3 / m2 ** 1.5 if m2 > 0 else np.nan

    return feats