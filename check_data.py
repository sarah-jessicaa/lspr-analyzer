from core.data_loader import iter_dataset, load_spectrum
from core.feature_extraction import extract_features

fail = 0
for path, label in iter_dataset():
    try:
        df = load_spectrum(path)
        feats = extract_features(df.wavelength.to_numpy(), df.intensity.to_numpy())
        print(f"{path.name:16s} {df.shape[0]:4d} titik  "
              f"peak={feats['peak_wavelength']:7.1f}  fwhm={feats['fwhm']:6.1f}")
    except Exception as e:
        fail += 1
        print(f"{path.name:16s} GAGAL: {e}")

print(f"\nselesai, {fail} file gagal")