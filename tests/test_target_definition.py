from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from databricks.modeling_utils import apply_binary_label, reject_target_leakage
from scripts.check_no_leakage import validate_threshold_contract


ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPTS = [
    ROOT / "databricks" / "04_train_logreg.py",
    ROOT / "databricks" / "05_train_tree_baseline.py",
    ROOT / "databricks" / "06_train_boosted_tree.py",
    ROOT / "databricks" / "09_train_xgboost.py",
    ROOT / "databricks" / "14_logreg_variable_selection.py",
]


def test_reject_target_leakage_blocks_target_columns() -> None:
    with pytest.raises(ValueError):
        reject_target_leakage(["age_years", "target_annual_claim_cost", "label"])


def test_reject_target_leakage_allows_prior_year_features() -> None:
    reject_target_leakage(["age_years", "prior_year_annual_claim_cost", "current_year_high_cost_indicator"])


def test_label_rule_uses_greater_than_or_equal_threshold() -> None:
    source = inspect.getsource(apply_binary_label)
    assert "target_annual_claim_cost" in source
    assert ">=" in source
    assert ">" not in source.replace(">=", "")


def test_training_scripts_do_not_compute_threshold_from_holdout_rows() -> None:
    for path in TRAINING_SCRIPTS:
        text = path.read_text()
        validate_threshold_contract(path)
        assert "compute_training_only_threshold(train_df, TARGET_QUANTILE)" in text, path.name
        assert "apply_threshold(train_df, threshold)" in text, path.name
        assert "reject_target_leakage" in text, path.name


def test_threshold_contract_rejects_combined_holdout_threshold_population(tmp_path: Path) -> None:
    leaking_script = tmp_path / "leaking_threshold.py"
    leaking_script.write_text(
        """
NUMERIC_FEATURES = ["age_years"]
CATEGORICAL_FEATURES = []

def add_training_target(train_df, validation_df, test_df):
    combined_df = train_df.unionByName(validation_df).unionByName(test_df)
    thresholds_df = combined_df.groupBy("target_year").agg(
        F.expr("percentile_approx(target_annual_claim_cost, 0.9)").alias("target_year_high_cost_threshold")
    )
    return combined_df.join(thresholds_df, "target_year")
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="combined train/validation/test|combined_df|direct threshold"):
        validate_threshold_contract(leaking_script)


def test_threshold_contract_rejects_direct_target_year_threshold_labeling(tmp_path: Path) -> None:
    leaking_script = tmp_path / "leaking_label.py"
    leaking_script.write_text(
        """
NUMERIC_FEATURES = ["age_years"]
CATEGORICAL_FEATURES = []

def add_training_target(train_df, validation_df, test_df):
    threshold = compute_training_only_threshold(train_df, TARGET_QUANTILE)
    labeled_train = train_df.withColumn(
        "label",
        (F.col("target_annual_claim_cost") >= F.col("target_year_high_cost_threshold")).cast("int"),
    )
    return labeled_train, apply_threshold(validation_df, threshold), apply_threshold(test_df, threshold), threshold
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="direct greater-than-or-equal label"):
        validate_threshold_contract(leaking_script)


def test_threshold_contract_requires_shared_training_only_helpers(tmp_path: Path) -> None:
    incomplete_script = tmp_path / "missing_helpers.py"
    incomplete_script.write_text(
        """
NUMERIC_FEATURES = ["age_years"]
CATEGORICAL_FEATURES = []

def add_training_target(train_df, validation_df, test_df):
    return train_df, validation_df, test_df, 0.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="compute_training_only_threshold"):
        validate_threshold_contract(incomplete_script)


def test_threshold_contract_accepts_training_only_shared_helper_pattern(tmp_path: Path) -> None:
    safe_script = tmp_path / "safe_threshold.py"
    safe_script.write_text(
        """
NUMERIC_FEATURES = ["age_years"]
CATEGORICAL_FEATURES = []

def add_training_target(train_df, validation_df, test_df):
    threshold = compute_training_only_threshold(train_df, TARGET_QUANTILE)
    return apply_threshold(train_df, threshold), apply_threshold(validation_df, threshold), apply_threshold(test_df, threshold), threshold
""",
        encoding="utf-8",
    )

    validate_threshold_contract(safe_script)
