#!/usr/bin/env python3
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from databricks.modeling_utils import reject_target_leakage

TRAINING_SCRIPTS = [
    ROOT / "databricks" / "04_train_logreg.py",
    ROOT / "databricks" / "05_train_tree_baseline.py",
    ROOT / "databricks" / "06_train_boosted_tree.py",
    ROOT / "databricks" / "09_train_xgboost.py",
    ROOT / "databricks" / "11_feature_audit.py",
    ROOT / "databricks" / "14_logreg_variable_selection.py",
]

FORBIDDEN_THRESHOLD_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "combined train/validation/test threshold population",
        re.compile(r"train_df\.unionByName\(validation_df\)\.unionByName\(test_df\)"),
    ),
    (
        "combined train/supplementary/test threshold population",
        re.compile(r"train_df\.unionByName\(supplementary_holdout_df\)\.unionByName\(test_df\)"),
    ),
    (
        "combined_df built from unionByName",
        re.compile(r"combined_df\s*=\s*.*unionByName", re.DOTALL),
    ),
    (
        "combined_df used for target-year threshold aggregation",
        re.compile(r"combined_df.*groupBy\([\"']target_year[\"']\).*agg", re.DOTALL),
    ),
    (
        "direct threshold computation outside shared utility",
        re.compile(r"percentile_approx\(\s*target_annual_claim_cost"),
    ),
    (
        "direct greater-than label against target_year_high_cost_threshold",
        re.compile(
            r"target_annual_claim_cost[^\n]{0,160}>\s*F\.col\([\"']target_year_high_cost_threshold[\"']\)"
        ),
    ),
    (
        "direct greater-than-or-equal label against target_year_high_cost_threshold",
        re.compile(
            r"target_annual_claim_cost[^\n]{0,160}>=\s*F\.col\([\"']target_year_high_cost_threshold[\"']\)"
        ),
    ),
]

REQUIRED_THRESHOLD_HELPERS = [
    "compute_training_only_threshold(train_df, TARGET_QUANTILE)",
    "apply_threshold(train_df, threshold)",
    "apply_threshold(test_df, threshold)",
]


def extract_list_assignment(path: Path, name: str) -> list[str]:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            values: list[str] = []
            for item in getattr(node.value, "elts", []):
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    values.append(item.value)
            return values
    return []


def validate_feature_contract(path: Path) -> None:
    features = extract_list_assignment(path, "NUMERIC_FEATURES") + extract_list_assignment(path, "CATEGORICAL_FEATURES")
    reject_target_leakage(features)


def validate_threshold_contract(path: Path) -> None:
    text = path.read_text()
    failures: list[str] = []

    for label, pattern in FORBIDDEN_THRESHOLD_PATTERNS:
        if pattern.search(text):
            failures.append(label)

    for helper in REQUIRED_THRESHOLD_HELPERS:
        if helper not in text:
            failures.append(f"missing required shared helper: {helper}")

    if failures:
        raise ValueError(f"{path.name}: target-threshold leakage contract failed: {'; '.join(failures)}")


def main() -> int:
    failures: list[str] = []
    for path in TRAINING_SCRIPTS:
        if not path.exists():
            continue
        try:
            validate_feature_contract(path)
            validate_threshold_contract(path)
        except ValueError as exc:
            failures.append(str(exc))

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    print("PASS: no target leakage feature columns or threshold-leakage patterns found in training scripts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
