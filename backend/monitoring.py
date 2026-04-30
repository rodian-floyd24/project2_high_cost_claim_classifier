from __future__ import annotations

import numpy as np

PSI_STABLE_MAX = 0.10
PSI_REVIEW_MIN = 0.25
CALIBRATION_GAP_ACCEPTABLE_MAX = 0.02
CALIBRATION_GAP_REVIEW_MIN = 0.05
TOP10_CAPTURE_WARNING_DROP = 0.10
TOP10_CAPTURE_REVIEW_DROP = 0.20


def population_stability_index(expected, actual, bins: int = 10) -> float:
    expected_values = np.asarray(expected, dtype=float)
    actual_values = np.asarray(actual, dtype=float)
    if expected_values.size == 0 or actual_values.size == 0:
        return 0.0
    breakpoints = np.quantile(expected_values, np.linspace(0.0, 1.0, bins + 1))
    breakpoints = np.unique(breakpoints)
    if breakpoints.size < 2:
        return 0.0
    expected_counts, _ = np.histogram(expected_values, bins=breakpoints)
    actual_counts, _ = np.histogram(actual_values, bins=breakpoints)
    expected_pct = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, None)
    actual_pct = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def drift_status(psi: float) -> str:
    if psi >= PSI_REVIEW_MIN:
        return "review_required"
    if psi >= PSI_STABLE_MAX:
        return "monitor"
    return "stable"


def calibration_status(abs_gap: float) -> str:
    if abs_gap >= CALIBRATION_GAP_REVIEW_MIN:
        return "review_required"
    if abs_gap >= CALIBRATION_GAP_ACCEPTABLE_MAX:
        return "warning"
    return "acceptable"


def top10_capture_status(baseline_capture: float, current_capture: float) -> str:
    if baseline_capture <= 0:
        return "insufficient_baseline"
    relative_drop = (baseline_capture - current_capture) / baseline_capture
    if relative_drop > TOP10_CAPTURE_REVIEW_DROP:
        return "review_required"
    if relative_drop >= TOP10_CAPTURE_WARNING_DROP:
        return "warning"
    return "acceptable"
