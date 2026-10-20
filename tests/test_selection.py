import pytest

from pulseguard.models.selection import choose_threshold, screening_utility


def test_utility_weights_auc_more_than_recall():
    assert screening_utility(1.0, 0.0) == pytest.approx(0.7)
    assert screening_utility(0.8, 0.6) == pytest.approx(0.7 * 0.8 + 0.3 * 0.6)


def test_utility_rejects_out_of_range():
    with pytest.raises(ValueError):
        screening_utility(1.2, 0.4)


def test_threshold_prefers_a_cutoff_inside_the_grid():
    y_true = [0, 0, 0, 1, 1, 1]
    probabilities = [0.1, 0.2, 0.35, 0.7, 0.8, 0.9]
    threshold = choose_threshold(y_true, probabilities)
    assert 0.2 <= threshold <= 0.8
