import pandas as pd
import pytest


@pytest.fixture
def clinical_row() -> dict:
    return {
        "age": 57,
        "sex": 1,
        "cp": 4,
        "trestbps": 140,
        "chol": 241,
        "fbs": 0,
        "restecg": 0,
        "thalach": 123,
        "exang": 1,
        "oldpeak": 1.2,
        "slope": 2,
        "ca": 0,
        "thal": 7,
        "source_site": "cleveland",
    }


@pytest.fixture
def tiny_training_frame(clinical_row) -> pd.DataFrame:
    rows = []
    for index in range(24):
        row = dict(clinical_row)
        row["age"] = 35 + index
        row["chol"] = 180 + index * 4
        row["oldpeak"] = 0.2 * (index % 6)
        row["exang"] = index % 2
        row["ca"] = index % 4
        row["heart_disease"] = int(index % 2 == 0)
        row["source_site"] = "cleveland" if index < 12 else "hungarian"
        rows.append(row)
    return pd.DataFrame(rows)
