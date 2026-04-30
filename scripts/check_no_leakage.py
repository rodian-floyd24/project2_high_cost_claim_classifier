#!/usr/bin/env python3
from __future__ import annotations

import ast
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


def main() -> int:
    for path in TRAINING_SCRIPTS:
        if not path.exists():
            continue
        features = extract_list_assignment(path, "NUMERIC_FEATURES") + extract_list_assignment(path, "CATEGORICAL_FEATURES")
        reject_target_leakage(features)
    print("PASS: no target leakage feature columns found in training scripts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
