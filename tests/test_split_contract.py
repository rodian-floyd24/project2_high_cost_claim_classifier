from __future__ import annotations

from pathlib import Path

import pytest

CANONICAL_SPLIT_VERSION = "xxhash64_bene_id_mod_100_v2_beneficiary_hash_holdout"

CONTRACT_FILES = [
    "README.md",
    "databricks/modeling_utils.py",
    "databricks/04_train_logreg.py",
    "databricks/05_train_tree_baseline.py",
    "databricks/06_train_boosted_tree.py",
    "databricks/07_model_comparison.py",
    "databricks/08_topk_capture_lift.py",
    "databricks/09_train_xgboost.py",
    "databricks/12_calibration_diagnostics.py",
    "backend/scoring.py",
    "backend/model_artifacts/model_metadata.json",
    "report_artifacts/final_results_table_test.csv",
    "report_artifacts/validation_packet/model_comparison_summary.csv",
]

FORBIDDEN_SPLIT_TEXT = [
    "xxhash64_bene_id_mod_100_v1",
    "temporal_target_year_holdout",
    "assign_temporal_hash_split",
    "split_gold_by_time",
    "target-year holdout",
    "Out-of-time validation",
]


@pytest.mark.parametrize("relative_path", CONTRACT_FILES)
def test_contract_files_use_canonical_split_version(relative_path: str) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / relative_path).read_text()
    assert CANONICAL_SPLIT_VERSION in text


def test_handoff_files_do_not_reintroduce_stale_split_language() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    searchable_roots = [
        repo_root / "README.md",
        repo_root / "databricks",
        repo_root / "backend",
        repo_root / "docs",
        repo_root / "report_artifacts",
    ]
    ignored = {
        repo_root / "report_artifacts" / "actuarial_blueprint_changes.patch",
        repo_root / "backend" / "model_artifacts" / "model" / "MLmodel",
        repo_root / "backend" / "model_artifacts" / "model" / "metadata" / "MLmodel",
    }

    offenders: list[str] = []
    for root in searchable_roots:
        paths = [root] if root.is_file() else [path for path in root.rglob("*") if path.is_file()]
        for path in paths:
            if path in ignored:
                continue
            if any(part.startswith(".") for part in path.relative_to(repo_root).parts):
                continue
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            for forbidden in FORBIDDEN_SPLIT_TEXT:
                if forbidden in text:
                    offenders.append(f"{path.relative_to(repo_root)} contains {forbidden!r}")

    assert offenders == []
