from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_repo_file(path: str) -> str:
    return (ROOT / path).read_text()


def test_downstream_diagnostics_use_selected_run_id_contract() -> None:
    downstream_files = [
        "databricks/08_topk_capture_lift.py",
        "databricks/12_calibration_diagnostics.py",
        "databricks/15_explainability_report.py",
        "databricks/16_model_monitoring.py",
    ]

    for path in downstream_files:
        text = read_repo_file(path)
        assert "selected_test_runs" in text, path
        assert "filter_to_selected_test_rows" in text, path
        assert "model_comparison_summary" not in text or path == "databricks/run_selection_utils.py"
        assert "agg(F.max(\"processed_at_utc\")" not in text, path
        assert "scores.agg(F.max(\"processed_at_utc\")" not in text, path


def test_monitoring_does_not_average_all_historical_prediction_scores() -> None:
    text = read_repo_file("databricks/16_model_monitoring.py")
    assert "current_scores.groupBy(\"model_name\", \"split_name\", \"run_id\")" in text
    assert "scores.groupBy(\"model_name\", \"split_name\")" not in text
    assert "current_selected_run" in text


def test_selected_run_id_is_carried_into_diagnostic_outputs() -> None:
    calibration_text = read_repo_file("databricks/12_calibration_diagnostics.py")
    explainability_text = read_repo_file("databricks/15_explainability_report.py")

    assert "T.StructField(\"run_id\", T.StringType(), False)" in calibration_text
    assert "selected_scores.select(\"model_name\", \"run_id\")" in calibration_text
    assert "latest.groupBy(\"model_name\", \"split_name\", \"run_id\")" in explainability_text


def test_model_comparison_selects_one_latest_run_id_with_timestamp_only_as_tiebreaker() -> None:
    text = read_repo_file("databricks/07_model_comparison.py")
    assert "groupBy(\"run_id\")" in text
    assert "row_number().over(run_window)" in text
    assert "selected_df.count() != 1" in text
    assert "processed_at_utc\") == F.lit(latest_ts)" not in text


def test_run_selection_utility_fails_when_selected_rows_are_missing() -> None:
    text = read_repo_file("databricks/run_selection_utils.py")
    assert "validate_one_run_per_model" in text
    assert "model_comparison_summary" in text
    assert "selected_test_runs" in text
    assert "filter_to_selected_test_rows" in text
    assert "missing rows for selected model run_ids" in text
