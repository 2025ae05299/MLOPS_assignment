import pandas as pd

from pulseguard.data.preprocess import apply_sentinel_rules, prepare_model_frame


def test_question_mark_and_zero_cholesterol_become_missing():
    raw = pd.DataFrame(
        [
            {
                "age": 50,
                "sex": 1,
                "cp": 4,
                "trestbps": 0,
                "chol": 0,
                "fbs": 0,
                "restecg": 0,
                "thalach": 0,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 2,
                "ca": None,
                "thal": None,
                "num": 1,
            }
        ]
    )
    cleaned = apply_sentinel_rules(raw)
    assert pd.isna(cleaned.loc[0, "chol"])
    assert pd.isna(cleaned.loc[0, "trestbps"])
    assert pd.isna(cleaned.loc[0, "thalach"])


def test_prepare_model_frame_requires_target(clinical_row):
    frame = pd.DataFrame([{**clinical_row, "heart_disease": 1}])
    features, target = prepare_model_frame(frame)
    assert list(features.columns)[-1] == "source_site"
    assert int(target.iloc[0]) == 1


def test_committed_table_contract():
    from pulseguard.data.preprocess import load_clean_frame

    frame = load_clean_frame()
    assert {"heart_disease", "source_site", "chol", "num"}.issubset(frame.columns)
    assert set(frame["heart_disease"].unique()).issubset({0, 1})
    assert frame["chol"].dropna().ge(50).all()
    assert len(frame) > 800
