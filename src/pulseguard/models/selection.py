"""Model-selection policy used by training and unit tests."""

from __future__ import annotations

from pulseguard.config import UTILITY_AUC_WEIGHT, UTILITY_RECALL_WEIGHT


def screening_utility(roc_auc: float, recall: float) -> float:
    """Blend ranking quality with sensitivity.

    The weights are an educational operating choice for this assignment:
    a missed positive is treated as more costly than a false alarm, but
    ROC-AUC still dominates so the model cannot win by predicting everyone positive.
    """
    if not 0.0 <= roc_auc <= 1.0:
        raise ValueError("roc_auc must be between 0 and 1")
    if not 0.0 <= recall <= 1.0:
        raise ValueError("recall must be between 0 and 1")
    return UTILITY_AUC_WEIGHT * float(roc_auc) + UTILITY_RECALL_WEIGHT * float(recall)


def choose_threshold(y_true, probabilities, beta: float = 1.5) -> float:
    """Pick a probability cutoff on out-of-fold scores using F-beta."""
    import numpy as np
    from sklearn.metrics import fbeta_score

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.20, 0.80, 61):
        predictions = (probabilities >= threshold).astype(int)
        score = fbeta_score(y_true, predictions, beta=beta, zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold
