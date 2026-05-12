from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

from backend.app import build_model_frame
from shared.feature_contract import (
    FEATURE_VERSION,
    MODEL_CATEGORICAL_FEATURES,
    MODEL_FEATURE_ORDER,
    MODEL_NUMERIC_FEATURES,
    RAW_INPUT_FIELDS,
)


ROOT = Path(__file__).resolve().parents[1]


def test_model_feature_order_is_canonical_concatenation() -> None:
    assert MODEL_FEATURE_ORDER == MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES
    assert len(MODEL_FEATURE_ORDER) == len(set(MODEL_FEATURE_ORDER))


def test_feature_contract_blocks_target_and_future_columns() -> None:
    forbidden_names = {
        "label",
        "target_annual_claim_cost",
        "target_high_cost_threshold",
        "target_year_high_cost_threshold",
    }
    assert not (set(MODEL_FEATURE_ORDER) & forbidden_names)
    assert not any(column.startswith(("target_", "next_year_")) for column in MODEL_FEATURE_ORDER)


def test_raw_input_fields_match_beneficiary_profile_schema() -> None:
    app_tree = ast.parse((ROOT / "backend" / "app.py").read_text())
    profile_class = next(
        node for node in app_tree.body if isinstance(node, ast.ClassDef) and node.name == "BeneficiaryProfile"
    )
    schema_fields = [
        node.target.id
        for node in profile_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]
    assert RAW_INPUT_FIELDS == schema_fields


def test_model_metadata_uses_shared_feature_version() -> None:
    metadata = json.loads((ROOT / "backend" / "model_artifacts" / "model_metadata.json").read_text())
    assert metadata["feature_version"] == FEATURE_VERSION


def test_training_scripts_do_not_define_independent_feature_lists() -> None:
    scripts = [
        ROOT / "databricks" / "04_train_logreg.py",
        ROOT / "databricks" / "05_train_tree_baseline.py",
        ROOT / "databricks" / "06_train_boosted_tree.py",
        ROOT / "databricks" / "09_train_xgboost.py",
        ROOT / "backend" / "train_local_model.py",
    ]
    guarded_names = {"NUMERIC_FEATURES", "CATEGORICAL_FEATURES", "CHRONIC_FLAG_FEATURES"}
    for script in scripts:
        tree = ast.parse(script.read_text())
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            assigned_names = {
                target.id for target in node.targets if isinstance(target, ast.Name) and target.id in guarded_names
            }
            if assigned_names:
                assert not isinstance(node.value, (ast.List, ast.Tuple, ast.Set)), (
                    f"{script} defines {sorted(assigned_names)} locally instead of importing the shared contract"
                )


def test_validation_reporting_and_serving_import_feature_contract() -> None:
    expected_importers = [
        ROOT / "databricks" / "03_gold.py",
        ROOT / "databricks" / "13_gold_pipeline_consistency_check.py",
        ROOT / "databricks" / "modeling_utils.py",
        ROOT / "backend" / "app.py",
        ROOT / "backend" / "rl" / "policy.py",
        ROOT / "backend" / "scoring.py",
        ROOT / "report_artifacts" / "generate_full_implementation_report.py",
        ROOT / "scripts" / "check_no_leakage.py",
    ]
    for path in expected_importers:
        assert "shared.feature_contract" in path.read_text(), f"{path} does not import the shared feature contract"


def test_model_frame_uses_artifact_signature_order_when_available() -> None:
    feature_row = {column: 0 for column in MODEL_FEATURE_ORDER}
    for column in MODEL_CATEGORICAL_FEATURES:
        feature_row[column] = "unknown"
    signature_columns = ["sex", "total_claim_count", "age_band", "enrollment_months_count"]
    dummy_model = SimpleNamespace(
        metadata=SimpleNamespace(
            signature=SimpleNamespace(
                inputs=SimpleNamespace(inputs=[SimpleNamespace(name=column) for column in signature_columns])
            )
        )
    )
    frame = build_model_frame(feature_row, dummy_model)
    assert list(frame.columns) == signature_columns
