#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "report_artifacts" / "validation_packet"
ARTIFACT_MAP = {
    "final_results_table_test.csv": "model_comparison_summary.csv",
    "xgboost_split_metrics.csv": "model_metrics_xgboost.csv",
    "topk_capture_curve_test.csv": "topk_capture_lift.csv",
    "xgboost_topk_curve_test.csv": "xgboost_topk_capture_lift.csv",
    "gold_feature_eda/missingness.csv": "feature_quality_missingness.csv",
    "gold_feature_eda/numeric_feature_quality.csv": "feature_quality_numeric.csv",
}
DOC_MAP = {
    "docs/model_card_template.md": "model_card.md",
    "docs/validation_report_template.md": "validation_report.md",
    "docs/monitoring_plan.md": "monitoring_plan.md",
}


def copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    shutil.copy2(source, destination)
    return True


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []
    missing: list[str] = []

    for source_name, destination_name in ARTIFACT_MAP.items():
        source = ROOT / "report_artifacts" / source_name
        if copy_if_exists(source, OUTPUT_DIR / destination_name):
            manifest.append(destination_name)
        else:
            missing.append(str(source.relative_to(ROOT)))

    for source_name, destination_name in DOC_MAP.items():
        source = ROOT / source_name
        if copy_if_exists(source, OUTPUT_DIR / destination_name):
            manifest.append(destination_name)
        else:
            missing.append(source_name)

    (OUTPUT_DIR / "README.md").write_text(
        "# Validation Packet\n\n"
        "Generated from local report artifacts and governance templates.\n\n"
        "## Included Files\n\n"
        + "".join(f"- {name}\n" for name in sorted(manifest))
        + "\n## Missing Source Artifacts\n\n"
        + ("".join(f"- {name}\n" for name in sorted(missing)) if missing else "- None\n")
    )
    print(f"validation packet written to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
