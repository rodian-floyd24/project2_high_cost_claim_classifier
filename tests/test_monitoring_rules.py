from __future__ import annotations

from backend.monitoring import calibration_status, drift_status, population_stability_index, top10_capture_status


def test_monitoring_status_thresholds() -> None:
    assert drift_status(0.05) == "stable"
    assert drift_status(0.12) == "monitor"
    assert drift_status(0.30) == "review_required"
    assert calibration_status(0.01) == "acceptable"
    assert calibration_status(0.03) == "warning"
    assert calibration_status(0.06) == "review_required"
    assert top10_capture_status(0.40, 0.30) == "review_required"


def test_population_stability_index_zero_for_same_distribution() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert population_stability_index(values, values) == 0.0
