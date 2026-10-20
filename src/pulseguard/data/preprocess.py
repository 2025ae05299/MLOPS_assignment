"""Cleaning helpers shared by acquisition, EDA, and tests."""

from __future__ import annotations

import pandas as pd

from pulseguard.config import DATA_PROCESSED, MODEL_INPUT_FIELDS, TARGET_COLUMN


def apply_sentinel_rules(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace UCI missing markers and physiologically impossible zeros with NA."""
    cleaned = frame.copy()
    numeric_columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "num",
    ]
    for column in numeric_columns:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    if "chol" in cleaned.columns:
        cleaned.loc[cleaned["chol"] <= 0, "chol"] = pd.NA
    if "trestbps" in cleaned.columns:
        cleaned.loc[cleaned["trestbps"] < 50, "trestbps"] = pd.NA
    if "thalach" in cleaned.columns:
        cleaned.loc[cleaned["thalach"] < 50, "thalach"] = pd.NA
    if "age" in cleaned.columns:
        cleaned.loc[(cleaned["age"] < 1) | (cleaned["age"] > 120), "age"] = pd.NA
    return cleaned


def load_clean_frame() -> pd.DataFrame:
    path = DATA_PROCESSED / "heart_disease_clean.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path)


def prepare_model_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model inputs and the binary educational target."""
    missing = [column for column in MODEL_INPUT_FIELDS + [TARGET_COLUMN] if column not in frame.columns]
    if missing:
        raise ValueError(f"Frame is missing required columns: {missing}")
    features = frame.loc[:, MODEL_INPUT_FIELDS].copy()
    target = frame[TARGET_COLUMN].astype(int)
    return features, target
