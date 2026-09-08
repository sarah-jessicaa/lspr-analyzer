import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

FEATURE_KEYS = [
    "peak_wavelength", "peak_intensity", "fwhm", "auc",
    "q_factor", "asymmetry",
    "intensity_450", "intensity_550", "intensity_650",
    "slope_400_500", "slope_500_600", "slope_600_700",
    "ratio_450_650", "ratio_550_650", "centroid_wavelength", "skewness",
]


def build_dataset(all_rows):
    df = pd.DataFrame(all_rows).dropna(subset=FEATURE_KEYS)
    X = df[FEATURE_KEYS].to_numpy(dtype=float)
    y = df["species"].to_numpy()
    return X, y, df


def _make_models(random_state):
    return {
        "LDA": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearDiscriminantAnalysis()),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=200, random_state=random_state)),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=6,
                random_state=random_state, eval_metric="mlogloss")),
        ]),
    }


@st.cache_data(show_spinner=False)
def analyze(all_rows, test_size=0.25, random_state=42, cv=5):
    X, y_str, df = build_dataset(all_rows)

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    labels = list(le.classes_)

    Xs = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(Xs)

    n_lda = min(2, len(labels) - 1)
    lda_vis = LinearDiscriminantAnalysis(n_components=n_lda)
    lda_coords = lda_vis.fit_transform(Xs, y)
    if lda_coords.ndim == 1 or lda_coords.shape[1] == 1:
        lda_coords = np.column_stack(
            [lda_coords.ravel(), np.zeros(len(lda_coords))])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y,
    )

    min_class = min(int(np.sum(y == c)) for c in np.unique(y))
    folds = max(2, min(cv, min_class))
    cv_split = StratifiedKFold(folds, shuffle=True, random_state=random_state)

    models = {}
    importance = {}
    for name, pipe in _make_models(random_state).items():
        scores = cross_val_score(pipe, X, y, cv=cv_split)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        models[name] = {
            "cv_mean": float(scores.mean()),
            "cv_std": float(scores.std()),
            "split_accuracy": float(accuracy_score(y_te, pred)),
            "confusion": confusion_matrix(
                y_te, pred, labels=list(range(len(labels)))),
        }
        step = pipe.named_steps["model"]
        if hasattr(step, "feature_importances_"):
            importance[name] = dict(
                zip(FEATURE_KEYS, step.feature_importances_))

    corr = np.corrcoef(X, rowvar=False)

    iso = IsolationForest(
        n_estimators=200, contamination=0.05,
        random_state=random_state)
    anom = iso.fit_predict(Xs)
    scores_iso = iso.decision_function(Xs)
    anomalies = [
        (fn, float(sc))
        for fn, a, sc in zip(df["filename"], anom, scores_iso)
        if a == -1
    ]

    return {
        "pca_coords": pca_coords,
        "lda_coords": lda_coords,
        "filenames": df["filename"].to_numpy(),
        "species": y_str,
        "explained": pca.explained_variance_ratio_,
        "models": models,
        "importance": importance,
        "labels": labels,
        "corr": corr,
        "anomalies": anomalies,
        "n_total": int(X.shape[0]),
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_te.shape[0]),
        "folds": folds,
    }