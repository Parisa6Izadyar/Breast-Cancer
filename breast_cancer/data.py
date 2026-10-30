import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

NUMERIC = ["Age", "Tumor Size", "Regional Node Examined", "Regional Node Positive"]
CATEGORICAL = ["Race", "Marital Status", "T Stage", "N Stage", "6th Stage",
               "Differentiation", "Grade", "A Stage", "Estrogen Status", "Progesterone Status"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Status"
EXCLUDED = "Survival Months"
ALIASES = {"Reginol Node Positive": "Regional Node Positive", "differentiate": "Differentiation"}


def normalise_columns(frame):
    frame = frame.copy()
    frame.columns = [ALIASES.get(str(c).strip(), str(c).strip()) for c in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names after normalization")
    return frame


def prepare_features(frame):
    frame = normalise_columns(frame)
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing predictor columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("The input contains no rows")
    features = frame[FEATURES].copy()
    for column in NUMERIC:
        features[column] = pd.to_numeric(features[column], errors="raise").astype(float)
        values = features[column].to_numpy()
        if np.isinf(values).any() or (values[~np.isnan(values)] < 0).any():
            raise ValueError(f"{column} must contain nonnegative finite values or missing values")
    for column in CATEGORICAL:
        values = features[column].astype("string").str.strip().replace("", pd.NA)
        features[column] = values.astype(object).where(values.notna(), np.nan)
    grades = {"I": "1", "II": "2", "III": "3", "IV": "4", "anaplastic; Grade IV": "4"}
    features["Grade"] = features["Grade"].replace(grades)
    invalid = set(features["Grade"].dropna()) - {"1", "2", "3", "4"}
    if invalid:
        raise ValueError(f"Unrecognized Grade values: {sorted(invalid)}")
    return features


def load_data(path: Path):
    if not path.is_file():
        raise ValueError(f"Input file not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("Use the Breast_Cancer.csv input; the archived XLSX has a different schema")
    raw = pd.read_csv(path)
    source = normalise_columns(raw)
    if TARGET not in source or source[TARGET].isna().any():
        raise ValueError("A complete Status column is required")
    labels = source[TARGET].astype(str).str.strip()
    if not set(labels).issubset({"Alive", "Dead"}) or labels.nunique() != 2:
        raise ValueError("Status must include both Alive and Dead, without other labels")
    frame = prepare_features(source)
    frame[TARGET] = labels.map({"Alive": 0, "Dead": 1}).to_numpy()
    if EXCLUDED in source:
        followup = pd.to_numeric(source[EXCLUDED], errors="raise").astype(float)
        if np.isinf(followup).any() or followup.dropna().lt(0).any():
            raise ValueError("Survival Months must be nonnegative and finite")
        frame[EXCLUDED] = followup.to_numpy()
    rows = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    groups = predictor_groups(frame)
    quality = {
        "source_file": path.name, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_rows": rows, "exact_duplicates_removed": rows - len(frame), "retained_rows": len(frame),
        "predictor_columns": FEATURES, "predictor_count": len(FEATURES),
        "excluded_predictors": [EXCLUDED], "unique_predictor_profiles": int(groups.nunique()),
        "repeated_predictor_profiles": int(groups.duplicated().sum()),
        "profiles_with_both_labels": int((frame.groupby(groups)[TARGET].nunique() > 1).sum()),
        "class_counts": {"Alive": int(frame[TARGET].eq(0).sum()), "Dead": int(frame[TARGET].eq(1).sum())},
        "missing_predictor_values": {c: int(frame[c].isna().sum()) for c in FEATURES},
    }
    return frame, quality


def predictor_groups(frame):
    # Group on pre-outcome predictors so identical profiles cannot cross a split.
    return pd.util.hash_pandas_object(prepare_features(frame), index=False).astype(str)


def grouped_splits(frame, folds=5, seed=42):
    if folds < 2 or frame[TARGET].value_counts().min() < folds:
        raise ValueError("Each class must have at least as many rows as the number of folds")
    groups = predictor_groups(frame)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = list(splitter.split(frame[FEATURES], frame[TARGET], groups))
    for train, test in result:
        if set(groups.iloc[train]) & set(groups.iloc[test]):
            raise ValueError("Predictor profiles overlap across partitions")
        if frame.iloc[train][TARGET].nunique() != 2 or frame.iloc[test][TARGET].nunique() != 2:
            raise ValueError("Every partition must contain both target classes")
    return result
