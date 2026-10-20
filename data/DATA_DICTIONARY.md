# Data dictionary

Source: Janosi, Steinbrunn, Pfisterer, and Detrano, Heart Disease, UCI Machine Learning Repository, https://doi.org/10.24432/C52P4X.

PulseGuard uses the four processed files, not the single Cleveland extract that most classroom notebooks copy.

| Column | Role | Notes |
| --- | --- | --- |
| age | numeric input | Years |
| sex | binary input | UCI coding: 0 female, 1 male |
| cp | categorical input | Chest-pain type 1-4 |
| trestbps | numeric input | Resting blood pressure. Values under 50 are treated as missing. |
| chol | numeric input | Serum cholesterol. Values at or below 0 are treated as unmeasured. This mainly affects Switzerland. |
| fbs | binary input | Fasting blood sugar > 120 mg/dl |
| restecg | categorical input | Resting ECG code 0, 1, 2 |
| thalach | numeric input | Maximum heart rate. Values under 50 are treated as missing. |
| exang | binary input | Exercise-induced angina |
| oldpeak | numeric input | ST depression induced by exercise relative to rest |
| slope | categorical input | Slope of the peak exercise ST segment. Missingness is retained as a flag. |
| ca | categorical input | Number of major vessels colored by fluoroscopy. Often missing outside Cleveland. |
| thal | categorical input | Thalassemia code, typically 3, 6, or 7. Often missing outside Cleveland. |
| num | severity label | Original UCI angiographic status, 0-4. Not used as a model input. |
| source_site | categorical input | cleveland, hungarian, switzerland, or va. Added by this project. |
| heart_disease | binary target | 1 when `num > 0`, else 0. Educational label only. |

Engineered fields created inside the saved pipeline, not accepted by the API:

- `st_burden`: oldpeak times slope
- `pressure_age_index`: trestbps times age / 100
- `hr_gap`: (220 - age) - thalach
- `ischemic_combo`: exercise angina and oldpeak of at least 1.0
- `age_band`: under 40, 40-54, 55-64, 65+
- missingness flags for cholesterol, ca, thal, and slope
