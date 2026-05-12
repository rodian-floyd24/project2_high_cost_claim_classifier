from __future__ import annotations

import json
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from shared.feature_contract import FEATURE_VERSION


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "report_artifacts"
MD_PATH = OUT_DIR / "full_project_implementation_report.md"
PDF_PATH = OUT_DIR / "full_project_implementation_report.pdf"

BLUE = "#183B56"
TEAL = "#1F8A8A"
GRAY = "#52616B"
LIGHT = "#F5F7FA"


def read_text(path: str, default: str = "") -> str:
    target = ROOT / path
    return target.read_text() if target.exists() else default


def add_page(pdf: PdfPages, title: str, blocks: list[str], footer: str = "") -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.07, 0.95, title, fontsize=19, fontweight="bold", color=BLUE, va="top")
    ax.plot([0.07, 0.93], [0.925, 0.925], color=TEAL, linewidth=2)

    y = 0.895
    for block in blocks:
        if block.startswith("## "):
            y -= 0.008
            ax.text(0.07, y, block[3:], fontsize=13, fontweight="bold", color=BLUE, va="top")
            y -= 0.035
            continue
        if block.startswith("- "):
            wrapped = textwrap.wrap(block[2:], width=92)
            for i, line in enumerate(wrapped):
                prefix = "• " if i == 0 else "  "
                ax.text(0.09, y, prefix + line, fontsize=10.2, color="#111827", va="top")
                y -= 0.022
            y -= 0.005
            continue
        wrapped = textwrap.wrap(block, width=96)
        for line in wrapped:
            ax.text(0.07, y, line, fontsize=10.4, color="#111827", va="top")
            y -= 0.022
        y -= 0.01

    if footer:
        ax.text(0.07, 0.035, footer, fontsize=8.5, color=GRAY, va="bottom")
    pdf.savefig(fig)
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, note: str = "") -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.04, 0.94, title, fontsize=18, fontweight="bold", color=BLUE, va="top")
    if note:
        ax.text(0.04, 0.89, note, fontsize=10, color=GRAY, va="top")
    table_ax = fig.add_axes([0.035, 0.08, 0.93, 0.76])
    table_ax.axis("off")
    table = table_ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="upper left",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D7DE")
        if row == 0:
            cell.set_facecolor(BLUE)
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F8FAFC")
    pdf.savefig(fig)
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: Path, caption: str) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.04, 0.94, title, fontsize=18, fontweight="bold", color=BLUE, va="top")
    ax.text(0.04, 0.89, caption, fontsize=10.2, color=GRAY, va="top")
    image = Image.open(image_path)
    image_ax = fig.add_axes([0.06, 0.08, 0.88, 0.76])
    image_ax.imshow(image)
    image_ax.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def format_results_table() -> pd.DataFrame:
    df = pd.read_csv(OUT_DIR / "final_results_table_test.csv")
    keep = [
        "model_name",
        "accuracy",
        "precision",
        "recall",
        "area_under_roc",
        "area_under_pr",
        "top_5_capture_rate",
        "top_10_capture_rate",
        "top_10_lift",
    ]
    df = df[keep].copy()
    df.columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "ROC-AUC",
        "PR-AUC",
        "Top 5% Capture",
        "Top 10% Capture",
        "Top 10% Lift",
    ]
    df["Model"] = df["Model"].str.replace("_", " ").str.title()
    df["Model"] = df["Model"].replace({"Xgboost": "XGBoost"})
    for column in df.columns[1:]:
        df[column] = df[column].map(lambda x: f"{x:.4f}")
    return df


def model_metric_row(model_name: str) -> pd.Series:
    df = pd.read_csv(OUT_DIR / "final_results_table_test.csv")
    matches = df[df["model_name"] == model_name]
    if matches.empty:
        raise ValueError(f"Missing model results for {model_name}")
    return matches.iloc[0]


def model_selection_sentence() -> str:
    gradient = model_metric_row("gradient_boosting")
    logistic = model_metric_row("logistic_regression")
    return (
        "Gradient boosting is selected as the primary operational model because it achieved "
        f"the strongest held-out PR-AUC ({gradient['area_under_pr']:.4f}) and competitive top-k capture. "
        f"Its ROC-AUC was {gradient['area_under_roc']:.4f}, top-10 capture was "
        f"{gradient['top_10_capture_rate']:.4f}, and top-10 lift was {gradient['top_10_lift']:.4f}. "
        "Logistic regression remains a highly competitive interpretable baseline, with top-10 capture "
        f"{logistic['top_10_capture_rate']:.4f}."
    )


def data_inventory_summary() -> pd.DataFrame:
    df = pd.read_csv(OUT_DIR / "project_data_inventory.csv")
    summary = df[["entity", "logical_name", "actual_row_count", "silver_table"]].copy()
    summary.columns = ["Entity", "Logical source", "Rows", "Silver table"]
    summary["Rows"] = summary["Rows"].map(lambda x: f"{int(x):,}")
    return summary


def markdown_report() -> str:
    model_meta = json.loads((ROOT / "backend/model_artifacts/model_metadata.json").read_text())
    results = format_results_table().to_markdown(index=False)
    inventory = data_inventory_summary().to_markdown(index=False)
    model_selection = model_selection_sentence()

    return f"""# Full Project Implementation Report

Generated: {datetime.now().strftime('%B %d, %Y %I:%M %p')}

## Executive Summary

This project builds an actuarial decision-support prototype for prospective Medicare high-cost risk prediction. CMS DE-SynPUF synthetic claims data are organized through a Databricks-style medallion pipeline into a beneficiary-year modeling table. Year-t features predict whether a beneficiary becomes top-decile high cost in year t+1. The final system includes supervised model comparison, top-k targeting diagnostics, calibration/governance artifacts, a FastAPI backend, a Streamlit frontend, and a simulated MDP/Q-learning policy layer.

The central methodological boundary is: the risk engine is empirically trained on observed data, while the reinforcement-learning policy layer is a simulated decision prototype built on stylized transition and reward assumptions.

## Repository Structure Implemented

- `data_ingestion/`: CMS raw file landing and manifest creation.
- `databricks/`: bronze, silver, gold, training, comparison, calibration, top-k, audit, explainability, and monitoring notebooks/scripts.
- `backend/`: FastAPI app, scoring schema, reason codes, monitoring helpers, model artifact, metadata, and RL policy layer.
- `frontend/`: Streamlit decision-support app with demo profiles.
- `docs/`: use-case, governance, monitoring, human-review, data-dictionary, model-card, and validation templates.
- `scripts/`: local checks, leakage checks, and validation packet generation.
- `tests/`: regression tests covering leakage, metrics, API schema, scoring, explanations, monitoring rules, and model metadata.
- `report_artifacts/`: final metrics, EDA artifacts, charts, validation packet, and generated reports.

## Data Sources and Ingestion

{inventory}

The raw files are staged into `object_storage/bronze`, extracted, inventoried, and registered into bronze and silver structures. Source grains differ by file type: beneficiary-year summary rows, inpatient/outpatient/carrier claim records, and prescription drug events. The gold layer reconciles these into one beneficiary-year row.

## Gold Table and Modeling Grain

The modeling grain is one row per `bene_id + year`. This is the correct annual risk-segmentation unit for prospective actuarial high-cost prediction. Gold-table checks enforce non-null keys, uniqueness, required columns, chronic-count validity, enrollment bounds, cost validity, and feature-version metadata.

EDA confirms 343,644 gold rows across 116,352 beneficiaries and calendar years 2008, 2009, and 2010. The prospective year-t to year-t+1 modeling frame contains 227,292 rows.

## Prospective Target and Leakage Control

Features are measured in beneficiary-year t. The target is measured from annual claim cost in year t+1. A beneficiary is labeled high cost when next-year annual claim cost is at or above the training-only top-decile threshold. This avoids same-year leakage and prevents validation/test target information from setting the threshold.

The final comparison split is the locked v2 beneficiary-hash holdout: `xxhash64_bene_id_mod_100_v2_beneficiary_hash_holdout`. Hash buckets `<15` define test, buckets `15-29` define validation, and buckets `>=30` define train.

Shared utilities in `databricks/modeling_utils.py` centralize prospective frame creation, split assignment, threshold application, and leakage rejection. The canonical feature contract lives in `shared/feature_contract.py`, and `scripts/check_no_leakage.py` validates the shared model feature order for target columns.

## Feature Engineering

The gold feature table includes demographic, enrollment, chronic-burden, utilization, provider-fragmentation, cost, log-cost, lagged-history, trend, and interaction features. Chronic-condition parsing was hardened so CMS chronic flags are not silently collapsed. The final data dictionary documents the beneficiary-year contract and key features.

## Supervised Modeling

The supervised layer compares four model families:

- Logistic regression: interpretable actuarial/statistical baseline.
- Random forest: nonlinear benchmark.
- Gradient boosting: primary operational model.
- XGBoost: high-recall challenger.

Logistic regression remains the interpretability anchor. Gradient boosting is used as the primary operational model because it produced the strongest held-out PR-AUC with competitive top-k capture.

## Final Held-Out Test Results

{results}

{model_selection}

## Top-K Operational Targeting

Top-k capture and lift translate model performance into care-management capacity terms. The project reports top-5% and top-10% capture/lift and includes full curve artifacts for operational review. These are more relevant than raw accuracy because the positive class is intentionally rare.

## Calibration, Monitoring, and Governance

Calibration diagnostics, monitoring thresholds, and governance files were added to frame the model as decision support. The documentation now includes intended use, prohibited uses, human-review policy, monitoring plan, model-card template, validation-report template, limitations, and a deployment runbook.

The model is not approved for autonomous coverage, benefit, pricing, reserving, clinical, or adverse consumer decisions.

## Backend API

The FastAPI backend exposes `/health`, `/metadata`, `/predict`, `/state`, `/recommend_action`, `/simulate`, and `/decision_support`. Responses include model name, model version, risk score, risk tier, operating action, reason codes, input-review flags, and human-review indicators.

The model serving contract is explicit:

```json
{json.dumps(model_meta, indent=2)}
```

Local Python 3.12 development is supported, but exact artifact-compatible serving should use Python 3.11 and `requirements-serving-py311.txt`.

This serving contract makes the deployed model auditable by tying predictions to a specific model version, feature version, split version, target definition, Python version, scikit-learn version, and MLflow run.

## Streamlit Frontend

The frontend app is `frontend/app.py` and runs with `./run_frontend.sh`. It provides three demo profiles: low-risk routine beneficiary, moderate chronic-care beneficiary, and very-high-risk complex beneficiary. It displays risk prediction, MDP state, recommended action, action comparison, and methodology limitations.

## Simulated Policy Layer

The MDP state includes risk tier, chronic burden, utilization intensity, and prior intervention status. The action set includes no action, low-touch outreach, care coordination call, and intensive case management. Tabular Q-learning is used to estimate action value in the simulated environment.

This is not causal treatment-effect learning. The CMS synthetic data do not contain real intervention histories, so recommendations are simulated operational decision support. Therefore, the policy recommendation should be interpreted as a prototype decision-support output, not as evidence that a given intervention causally reduces future cost.

## Validation and Tests

The final verification stack includes:

- `python3 -m pip install -r requirements-dev.txt`
- `python3 -m compileall databricks backend tests scripts test_project.py report_artifacts`
- `pytest`
- `./scripts/run_local_tests.sh`
- `python3 test_project.py`
- Backend `/health` and `/metadata` smoke checks.
- Streamlit frontend reachability check.

Current local result: 33 passed, 1 skipped. The skipped test is the artifact reproduction check that only enforces sklearn 1.3.0 when running under the artifact's Python 3.11 line.

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-dev.txt
./run_backend.sh
./run_frontend.sh
```

Open `http://localhost:8501` for the app. The backend runs at `http://127.0.0.1:8000`.

## Limitations

- CMS DE-SynPUF is synthetic and historical, not a live production population.
- The model predicts high-cost status, not causal savings from intervention.
- The RL layer is simulated and should not be interpreted as validated clinical or operational intervention effectiveness.
- Exact model-artifact reproduction requires the Python 3.11 / sklearn 1.3.0 serving environment.

## Final Interpretation

This project is best understood as a governed actuarial risk-ranking and decision-support pipeline, not merely a classifier. It connects raw claims data to a prospective target, validates a stable beneficiary-year modeling grain, compares supervised models on operational metrics, exposes model metadata and reason codes through an API, presents outputs through a Streamlit app, and documents the governance controls needed for responsible actuarial use.
"""


def make_pdf() -> None:
    results = format_results_table()
    inventory = data_inventory_summary()
    model_meta = json.loads((ROOT / "backend/model_artifacts/model_metadata.json").read_text())
    gradient = model_metric_row("gradient_boosting")
    logistic = model_metric_row("logistic_regression")

    with PdfPages(PDF_PATH) as pdf:
        add_page(
            pdf,
            "Actuarial Decision-Support Prototype",
            [
                "A complete start-to-finish implementation report for the high-cost Medicare beneficiary risk project.",
                "This project converts CMS DE-SynPUF synthetic claims into a governed prospective risk-ranking system with supervised ML, validation artifacts, FastAPI serving, Streamlit presentation, and a simulated policy layer.",
                "Core framing: this is a prospective high-cost risk-ranking system, not a same-year cost classifier and not a causal treatment recommendation system.",
                f"Generated {datetime.now().strftime('%B %d, %Y at %I:%M %p')}.",
            ],
            footer="Project 2: Distributed Systems for Data Science",
        )
        add_page(
            pdf,
            "1. What Was Built",
            [
                "## End-to-end system",
                "- Raw CMS files are landed into project object storage and inventoried.",
                "- Databricks-style bronze, silver, and gold scripts transform raw source files into a beneficiary-year modeling table.",
                "- A prospective modeling frame predicts year t+1 high-cost status from year t features.",
                "- Logistic regression, random forest, gradient boosting, and XGBoost are compared on held-out test data.",
                "- Gradient boosting is used as the primary operational model because it produced the strongest balanced ranking performance.",
                "- FastAPI serves risk scores, reason codes, metadata, MDP state, recommendations, and simulation comparisons.",
                "- Streamlit provides an interactive decision-support demo with three scenario presets.",
                "- Governance files document intended use, prohibited use, monitoring, validation, human review, and limitations.",
            ],
        )
        add_table_page(
            pdf,
            "2. Source Data Inventory",
            inventory,
            "CMS DE-SynPUF source files are staged and standardized before aggregation to beneficiary-year gold features.",
        )
        add_page(
            pdf,
            "3. Data Pipeline and Gold Grain",
            [
                "## Medallion architecture",
                "- Bronze stores raw landed and extracted CMS source records.",
                "- Silver standardizes beneficiary, inpatient, outpatient, carrier, and prescription drug event tables.",
                "- Gold aggregates all sources into one row per beneficiary-year.",
                "## Gold contract",
                "- Primary key: bene_id + year.",
                f"- Feature version: {FEATURE_VERSION}.",
                "- Blocking failures: duplicate beneficiary-year keys, missing keys, empty row count, invalid enrollment months, invalid chronic-condition counts, and negative component costs.",
                "## EDA confirmation",
                "- Gold rows: 343,644.",
                "- Distinct beneficiaries: 116,352.",
                "- Prospective modeling rows: 227,292.",
                "- Duplicate bene_id + year rows: 0.",
            ],
        )
        for title, path, caption in [
            (
                "4. Chronic Burden Signal",
                OUT_DIR / "gold_feature_eda/median_cost_by_chronic_count.png",
                "Median annual claim cost rises with chronic-condition count, supporting chronic burden features and nonlinear terms.",
            ),
            (
                "5. Target Balance by Year",
                OUT_DIR / "gold_feature_eda/target_balance_by_year.png",
                "The top-decile target creates an intentionally imbalanced but operationally meaningful high-cost label.",
            ),
            (
                "6. Feature Signal Checks",
                OUT_DIR / "gold_feature_eda/top_feature_label_correlations.png",
                "Utilization intensity, provider count, chronic burden, and cost features show the strongest prospective signal.",
            ),
        ]:
            if path.exists():
                add_image_page(pdf, title, path, caption)
        add_page(
            pdf,
            "7. Prospective Target and Leakage Controls",
            [
                "Features are measured in beneficiary-year t. The high-cost label is computed from annual claim cost in year t+1.",
                "The final comparison split is the locked v2 beneficiary-hash holdout: xxhash64_bene_id_mod_100_v2_beneficiary_hash_holdout.",
                "Hash buckets <15 define test, buckets 15-29 define validation, and buckets >=30 define train.",
                "The high-cost threshold is computed from the training split only. Validation and test costs do not set the threshold.",
                "The leakage rule blocks target_annual_claim_cost, target threshold fields, label columns, and next-year features from model input lists.",
                "Shared utilities in databricks/modeling_utils.py centralize the modeling frame, split assignment, threshold application, and leakage validation.",
                "scripts/check_no_leakage.py statically checks training scripts for forbidden target columns.",
            ],
        )
        add_table_page(
            pdf,
            "8. Final Held-Out Test Results",
            results,
            "Gradient boosting is selected for operational ranking because it has the strongest held-out PR-AUC and a competitive top-k capture profile.",
        )
        for title, path, caption in [
            (
                "9. Top-K Capture Curve",
                OUT_DIR / "topk_capture_curve_test.png",
                "Top-k capture shows how many true high-cost beneficiaries are captured when operations can only review the highest-risk fraction.",
            ),
            (
                "10. Top-K Lift Curve",
                OUT_DIR / "topk_lift_curve_test.png",
                "Lift translates ranking quality into operational concentration versus random selection.",
            ),
        ]:
            if path.exists():
                add_image_page(pdf, title, path, caption)
        add_page(
            pdf,
            "11. Model Selection Interpretation",
            [
                "- Logistic regression remains the interpretable statistical baseline.",
                (
                    f"- Gradient boosting achieved ROC-AUC {gradient['area_under_roc']:.4f}, "
                    f"PR-AUC {gradient['area_under_pr']:.4f}, and top-10 capture "
                    f"{gradient['top_10_capture_rate']:.4f} on the held-out test set."
                ),
                "- Random forest was competitive on top-k capture but weaker on discrimination.",
                "- XGBoost achieved the highest recall but lower precision, lower accuracy, and weaker top-k balance, so it is a high-sensitivity alternative rather than the primary balanced model.",
                (
                    f"- Gradient boosting has a top-10% lift of {gradient['top_10_lift']:.4f}; "
                    "logistic regression remains a highly competitive interpretable baseline "
                    f"with top-10 capture {logistic['top_10_capture_rate']:.4f}."
                ),
                "- Accuracy is reported but is not the primary selection criterion because the positive class is intentionally rare.",
            ],
        )
        add_page(
            pdf,
            "12. API and Serving Layer",
            [
                "## FastAPI endpoints",
                "- GET /health: model and Q-table load health plus serving metadata.",
                "- GET /metadata: model metadata, target definition, operating policy, required fields, and RL policy metadata.",
                "- POST /predict: supervised risk output with model version, risk score, risk tier, recommended action, review flags, and reason codes.",
                "- POST /decision_support: consolidated prediction, MDP state, recommendation, and action simulation.",
                "## Serving contract",
                json.dumps(model_meta, indent=2),
                "This serving contract makes the deployed model auditable by tying predictions to a specific model version, feature version, split version, target definition, Python version, scikit-learn version, and MLflow run.",
            ],
        )
        add_page(
            pdf,
            "13. Streamlit Frontend and Demo",
            [
                "The frontend runs from frontend/app.py through ./run_frontend.sh and is available at http://localhost:8501.",
                "The app includes three scenario presets: low-risk routine beneficiary, moderate chronic-care beneficiary, and very-high-risk complex beneficiary.",
                "The frontend displays risk prediction, MDP state, recommended action, action-by-action comparison, and a visible methodology/limitation block.",
                "During demos, repeat: the risk engine is empirically trained on observed data, while the reinforcement-learning policy layer is a simulated decision prototype built on stylized transition and reward assumptions.",
            ],
        )
        add_page(
            pdf,
            "14. Simulated MDP/Q-Learning Policy Layer",
            [
                "The policy layer maps supervised risk into an operational MDP state.",
                "State dimensions are risk tier, chronic burden, utilization intensity, and prior intervention status.",
                "Actions are no action, low-touch outreach, care coordination call, and intensive case management.",
                "Tabular Q-learning estimates the highest-value action under stylized transition and reward assumptions.",
                "This is not causal intervention modeling. It is a decision-support prototype that demonstrates how risk scores could feed operational recommendations.",
                "Therefore, the policy recommendation should be interpreted as a prototype decision-support output, not as evidence that a given intervention causally reduces future cost.",
            ],
        )
        add_page(
            pdf,
            "15. Governance and Methodological Controls",
            [
                "- Intended use: care-management queue prioritization, actuarial risk segmentation, and experience-study support.",
                "- Prohibited use: autonomous coverage, benefit, pricing, reserving, clinical, or adverse consumer decisions.",
                "- Human review: required for material operational use and flagged profiles.",
                "- Monitoring: data quality, feature drift, prediction drift, calibration, top-k capture, and operational override monitoring.",
                "- Runtime contract: Python 3.11.10 and scikit-learn 1.3.0 for exact artifact reproduction.",
                "- Documentation artifacts: model card template, validation report template, monitoring plan, human-review policy, data dictionary, governance policy, limitations, and deployment runbook.",
            ],
        )
        add_page(
            pdf,
            "16. Verification Completed",
            [
                "The final local verification sequence was run after submission-control edits.",
                "- python3 -m pip install -r requirements-dev.txt: completed with requirements already satisfied.",
                "- python3 -m compileall databricks backend tests scripts test_project.py report_artifacts: passed.",
                "- pytest: 33 passed, 1 skipped.",
                "- ./scripts/run_local_tests.sh: passed and leakage check passed.",
                "- python3 test_project.py: PASS.",
                "- Backend /health and /metadata smoke checks: passed.",
                "- Streamlit frontend reachability check: passed.",
                "The skipped test is expected under Python 3.12 because exact artifact reproduction is tied to the Python 3.11 / scikit-learn 1.3.0 serving contract.",
            ],
        )
        add_page(
            pdf,
            "17. How to Reproduce",
            [
                "## Local development",
                "python3 -m venv .venv",
                "source .venv/bin/activate",
                "python3 -m pip install --upgrade pip",
                "python3 -m pip install -r requirements-dev.txt",
                "./run_backend.sh",
                "./run_frontend.sh",
                "## Tests",
                "python3 -m compileall databricks backend tests scripts test_project.py report_artifacts",
                "pytest",
                "./scripts/run_local_tests.sh",
                "python3 test_project.py",
                "## Artifact-compatible serving",
                "python3.11 -m pip install -r requirements-serving-py311.txt",
                "python3.11 -m uvicorn backend.app:app --host 127.0.0.1 --port 8000",
            ],
        )
        add_page(
            pdf,
            "18. Final Takeaway",
            [
                "The project is a governed actuarial risk-ranking and decision-support pipeline.",
                "It starts with raw CMS synthetic claims, hardens a medallion data pipeline, builds a prospective beneficiary-year target, compares supervised models using operational metrics, serves a versioned model through FastAPI, presents decisions through Streamlit, and documents the controls needed for responsible use.",
                "The correct one-sentence framing is: this is a prospective high-cost risk-ranking system, not a same-year cost classifier and not a causal treatment recommendation system.",
            ],
        )


def main() -> None:
    MD_PATH.write_text(markdown_report())
    make_pdf()
    print(f"wrote {MD_PATH}")
    print(f"wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
