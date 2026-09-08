import re
from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_spectrum(source):
    if hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        lines = raw.splitlines()
    else:
        with open(source, "r", errors="ignore") as f:
            lines = f.readlines()

    rows = []
    for line in lines:
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    if not rows:
        raise ValueError(
            "no numeric wavelength/intensity pairs found in file "
            "(expected two columns separated by whitespace or commas)"
        )
    df = pd.DataFrame(rows, columns=["wavelength", "intensity"])
    return df.sort_values("wavelength").reset_index(drop=True)


def parse_label(filename):
    # format file: sapi1%-1.txt
    m = re.match(r"([a-z]+)(\d+)%-(\d+)", filename.lower())
    if not m:
        return None
    return {
        "species": m.group(1),
        "concentration": int(m.group(2)),
        "replicate": int(m.group(3)),
    }


def iter_dataset(data_dir="data"):
    for path in sorted(Path(data_dir).rglob("*.txt")):
        label = parse_label(path.name)
        if label is None:
            continue
        yield path, label