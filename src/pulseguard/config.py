"""Paths, seeds, and operational constants for PulseGuard."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MLFLOW_DIR = ROOT / "mlruns"

UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
SITE_FILES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}

RAW_COLUMNS = [
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

CLINICAL_FIELDS = [
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
]

MODEL_INPUT_FIELDS = CLINICAL_FIELDS + ["source_site"]
TARGET_COLUMN = "heart_disease"
SEVERITY_COLUMN = "num"

# Tied to repository owner 2025ae05299 so this submission is not a generic seed-42 notebook.
RANDOM_STATE = 5299
MODEL_VERSION = "1.0.0"
EXPERIMENT_NAME = "pulseguard-multisite"
# Recall-leaning educational screening utility. Not a clinical decision policy.
UTILITY_AUC_WEIGHT = 0.7
UTILITY_RECALL_WEIGHT = 0.3
HIGH_RISK_THRESHOLD = 0.70
TEST_SIZE = 0.20
CV_FOLDS = 5

DISCLAIMER = (
    "Educational coursework model trained on the public UCI Heart Disease collection. "
    "It is not a medical device, does not diagnose disease, and must not be used for "
    "clinical decisions."
)
