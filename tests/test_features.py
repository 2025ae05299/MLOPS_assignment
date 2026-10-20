import numpy as np
import pandas as pd

from pulseguard.features.builder import OUTPUT_COLUMNS, PulseGuardFeatureBuilder


def test_feature_builder_flags_missing_vessel_count(clinical_row):
    row = dict(clinical_row)
    row["ca"] = np.nan
    row["chol"] = 0
    built = PulseGuardFeatureBuilder().transform(pd.DataFrame([row]))
    assert list(built.columns) == OUTPUT_COLUMNS
    assert built.loc[0, "ca"] == "unknown"
    assert built.loc[0, "ca_missing"] == 1
    assert built.loc[0, "chol_missing"] == 1
    assert np.isnan(built.loc[0, "chol"])


def test_ischemic_combo_and_age_band(clinical_row):
    row = dict(clinical_row)
    row["age"] = 67
    row["exang"] = 1
    row["oldpeak"] = 1.4
    built = PulseGuardFeatureBuilder().transform(pd.DataFrame([row]))
    assert built.loc[0, "ischemic_combo"] == 1
    assert built.loc[0, "age_band"] == "65_plus"
    assert built.loc[0, "st_burden"] == row["oldpeak"] * row["slope"]
