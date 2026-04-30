from __future__ import annotations

from databricks.metric_utils import calibration_by_decile, calibration_summary, classification_metrics, topk_metrics


def test_classification_and_topk_metrics() -> None:
    y_true = [0, 0, 1, 1, 1]
    y_prob = [0.05, 0.2, 0.6, 0.8, 0.9]

    metrics = classification_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["row_count"] == 5
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["pr_auc"] >= metrics["positive_rate"]

    topk = topk_metrics(y_true, y_prob, [0.4])[0]
    assert topk["selected_count"] == 2
    assert topk["captured_positive_count"] == 2
    assert topk["lift_at_k"] > 1.0


def test_calibration_outputs() -> None:
    y_true = [0, 0, 1, 1, 1, 0, 1, 0, 1, 0]
    y_prob = [0.05, 0.1, 0.6, 0.7, 0.9, 0.2, 0.8, 0.3, 0.75, 0.4]
    summary = calibration_summary(y_true, y_prob)
    deciles = calibration_by_decile(y_true, y_prob)

    assert summary["row_count"] == 10
    assert "absolute_calibration_gap" in summary
    assert len(deciles) == 10
