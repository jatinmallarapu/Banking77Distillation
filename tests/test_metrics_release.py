"""Check calibration edge cases, regression tolerances and incomplete-release rejection."""

import numpy as np
import pytest

from b77.gate import regressed
from b77.metrics import calibration, choose_threshold, evaluate
from b77.release import readiness


def test_perfect_confidence_is_in_last_bin():
    result = calibration(np.array([0, 1]), np.array([[1, 0], [0, 1]]))
    assert result["ece"] == 0 and result["bins"][-1]["count"] == 2


def test_no_reliable_coverage_abstains_everywhere():
    selected = choose_threshold(np.array([1] * 40), np.tile([0.9, 0.1], (40, 1)))
    assert selected["threshold"] > 1 and selected["validation_coverage"] == 0


def test_metrics_and_intervals_have_real_denominators():
    rows = [{"id": str(i), "label": i % 2} for i in range(20)]
    probability = np.array([[0.9, 0.1] if r["label"] == 0 else [0.1, 0.9] for r in rows])
    result = evaluate(rows, probability, ["a", "b"], 0, [1.0] * 20, bootstrap=10)
    assert result["accuracy"] == 1 and result["macro_f1_ci95"] == [1, 1]


def test_nonprobabilities_fail():
    with pytest.raises(ValueError):
        evaluate(
            [{"id": "a", "label": 0}], np.array([[2.0, -1.0]]), ["a", "b"], 0, [1.0], bootstrap=2
        )


def test_gate_is_one_absolute_percentage_point():
    assert not regressed(0.90, 0.89)
    assert regressed(0.90, 0.889)


def test_incomplete_training_cannot_create_a_release(tmp_path):
    with pytest.raises(ValueError, match="Missing completed"):
        readiness(tmp_path)
