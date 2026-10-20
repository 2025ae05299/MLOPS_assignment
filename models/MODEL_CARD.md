# PulseGuard model card

- Model: `hist_gradient_boosting`
- Version: `1.0.0`
- Trained at: 2026-10-20T12:05:54.276733+00:00
- Random seed: 5299
- Decision threshold: 0.210
- High-risk band: probability >= 0.70

## Held-out metrics

- Accuracy: 0.815
- Precision: 0.770
- Recall: 0.951
- F1: 0.851
- ROC-AUC: 0.911
- PR-AUC: 0.913

## Selection policy

Maximize 0.70 * OOF ROC-AUC + 0.30 * OOF recall on the train split. Threshold tuned by F1.5 on OOF scores. Test split used once.

## Intended use

Educational coursework model trained on the public UCI Heart Disease collection. It is not a medical device, does not diagnose disease, and must not be used for clinical decisions.

Inputs are the 13 UCI clinical fields plus an optional `source_site`.
The saved object is a scikit-learn Pipeline, so scaling and encoding travel with the model.
