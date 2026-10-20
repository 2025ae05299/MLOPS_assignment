"""Sklearn transformer that keeps training and serving transformations identical."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pulseguard.data.preprocess import apply_sentinel_rules

OUTPUT_COLUMNS = [
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
    "source_site",
    "st_burden",
    "pressure_age_index",
    "hr_gap",
    "ischemic_combo",
    "age_band",
    "chol_missing",
    "ca_missing",
    "thal_missing",
    "slope_missing",
]

NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "st_burden",
    "pressure_age_index",
    "hr_gap",
]
BINARY_FEATURES = [
    "sex",
    "fbs",
    "exang",
    "ischemic_combo",
    "chol_missing",
    "ca_missing",
    "thal_missing",
    "slope_missing",
]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "ca", "thal", "source_site", "age_band"]


def _code(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return "unknown"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


class PulseGuardFeatureBuilder(BaseEstimator, TransformerMixin):
    """Build screening features and explicit missingness flags.

    Missing vessel count and thalassemia codes are kept as their own category
    instead of being mode-imputed. Those fields are missing much more often at
    the Switzerland and VA sites, so a Cleveland mode would hide a real shift.
    """

    def fit(self, X, y=None):  # noqa: N803 - sklearn API
        return self

    def transform(self, X):  # noqa: N803 - sklearn API
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if "source_site" not in frame.columns:
            frame["source_site"] = "unspecified"
        frame = apply_sentinel_rules(frame)

        for column in ["oldpeak", "slope", "exang", "age", "trestbps", "thalach", "chol"]:
            if column not in frame.columns:
                frame[column] = np.nan

        frame["chol_missing"] = frame["chol"].isna().astype(int)
        frame["ca_missing"] = frame["ca"].isna().astype(int)
        frame["thal_missing"] = frame["thal"].isna().astype(int)
        frame["slope_missing"] = frame["slope"].isna().astype(int)
        frame["st_burden"] = frame["oldpeak"].fillna(0) * frame["slope"].fillna(1)
        frame["pressure_age_index"] = frame["trestbps"] * frame["age"] / 100.0
        frame["hr_gap"] = (220 - frame["age"]) - frame["thalach"]
        frame["ischemic_combo"] = (
            (frame["exang"].fillna(0) >= 1) & (frame["oldpeak"].fillna(0) >= 1.0)
        ).astype(int)
        frame["age_band"] = pd.cut(
            frame["age"],
            bins=[0, 40, 55, 65, 120],
            labels=["under_40", "40_54", "55_64", "65_plus"],
        ).astype(str)
        frame.loc[frame["age"].isna(), "age_band"] = "unknown"

        for column in CATEGORICAL_FEATURES:
            if column == "age_band":
                continue
            frame[column] = frame[column].map(_code) if column in frame.columns else "unknown"
        frame["source_site"] = frame["source_site"].fillna("unspecified").astype(str)
        numeric_out = [
            "age",
            "sex",
            "trestbps",
            "chol",
            "fbs",
            "thalach",
            "exang",
            "oldpeak",
            "st_burden",
            "pressure_age_index",
            "hr_gap",
            "ischemic_combo",
            "chol_missing",
            "ca_missing",
            "thal_missing",
            "slope_missing",
        ]
        for column in numeric_out:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.loc[:, OUTPUT_COLUMNS].reset_index(drop=True)

    def get_feature_names_out(self, input_features=None):  # noqa: ARG002
        return np.asarray(OUTPUT_COLUMNS, dtype=object)


def build_pipeline(estimator) -> Pipeline:
    """Full reproducible scorer: engineered features, imputation, encoding, model."""
    builder = PulseGuardFeatureBuilder()
    builder.set_output(transform="pandas")
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "bin",
                SimpleImputer(strategy="most_frequent"),
                BINARY_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("features", builder),
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )
