# Full Project Implementation Report

Generated: April 30, 2026 10:48 AM

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

| Entity                   | Logical source       |      Rows | Silver table             |
|:-------------------------|:---------------------|----------:|:-------------------------|
| beneficiary_summary      | beneficiary_2008     |   116,352 | silver_beneficiaries     |
| beneficiary_summary      | beneficiary_2009     |   114,538 | silver_beneficiaries     |
| beneficiary_summary      | beneficiary_2010     |   112,754 | silver_beneficiaries     |
| inpatient_claims         | inpatient_2008_2010  |    66,773 | silver_inpatient_claims  |
| outpatient_claims        | outpatient_2008_2010 |   790,790 | silver_outpatient_claims |
| carrier_claims           | carrier_2008_2010_a  | 2,370,667 | silver_carrier_claims    |
| carrier_claims           | carrier_2008_2010_b  | 2,370,668 | silver_carrier_claims    |
| prescription_drug_events | pde_2008_2010        | 5,552,421 | silver_pde               |

The raw files are staged into `object_storage/bronze`, extracted, inventoried, and registered into bronze and silver structures. Source grains differ by file type: beneficiary-year summary rows, inpatient/outpatient/carrier claim records, and prescription drug events. The gold layer reconciles these into one beneficiary-year row.

## Gold Table and Modeling Grain

The modeling grain is one row per `bene_id + year`. This is the correct annual risk-segmentation unit for prospective actuarial high-cost prediction. Gold-table checks enforce non-null keys, uniqueness, required columns, chronic-count validity, enrollment bounds, cost validity, and feature-version metadata.

EDA confirms 343,644 gold rows across 116,352 beneficiaries and calendar years 2008, 2009, and 2010. The prospective year-t to year-t+1 modeling frame contains 227,292 rows.

## Prospective Target and Leakage Control

Features are measured in beneficiary-year t. The target is measured from annual claim cost in year t+1. A beneficiary is labeled high cost when next-year annual claim cost is at or above the training-only top-decile threshold. This avoids same-year leakage and prevents validation/test target information from setting the threshold.

Shared utilities in `databricks/modeling_utils.py` centralize prospective frame creation, split assignment, threshold application, and leakage rejection. `scripts/check_no_leakage.py` statically checks training feature lists for target columns.

## Feature Engineering

The gold feature table includes demographic, enrollment, chronic-burden, utilization, provider-fragmentation, cost, log-cost, lagged-history, trend, and interaction features. Chronic-condition parsing was hardened so CMS chronic flags are not silently collapsed. The final data dictionary documents the beneficiary-year contract and key features.

## Supervised Modeling

The supervised layer compares four model families:

- Logistic regression: interpretable actuarial/statistical baseline.
- Random forest: nonlinear benchmark.
- Gradient boosting: primary operational model.
- XGBoost: high-recall challenger.

Logistic regression remains the interpretability anchor. Gradient boosting is used as the primary operational model because it produced the strongest held-out ranking performance by PR-AUC and top-k capture.

## Final Held-Out Test Results

| Model               |   Accuracy |   Precision |   Recall |   ROC-AUC |   PR-AUC |   Top 5% Capture |   Top 10% Capture |   Top 10% Lift |
|:--------------------|-----------:|------------:|---------:|----------:|---------:|-----------------:|------------------:|---------------:|
| Logistic Regression |     0.8603 |      0.3432 |   0.4469 |    0.806  |   0.3634 |           0.2386 |            0.3799 |         3.7994 |
| Gradient Boosting   |     0.8533 |      0.3329 |   0.4769 |    0.8099 |   0.3879 |           0.2511 |            0.3877 |         3.8767 |
| Random Forest       |     0.8525 |      0.3207 |   0.4747 |    0.803  |   0.3634 |           0.2493 |            0.3819 |         3.8192 |
| XGBoost             |     0.7131 |      0.2179 |   0.7145 |    0.7989 |   0.3617 |           0.2322 |            0.3691 |         3.6897 |

Gradient boosting achieved ROC-AUC 0.8099, PR-AUC 0.3879, top-10 capture 38.77%, and top-10 lift 3.8767. The top-10 lift means the highest-risk 10% of beneficiaries contains about 3.88 times as many future high-cost cases as would be expected under random selection.

## Top-K Operational Targeting

Top-k capture and lift translate model performance into care-management capacity terms. The project reports top-5% and top-10% capture/lift and includes full curve artifacts for operational review. These are more relevant than raw accuracy because the positive class is intentionally rare.

## Calibration, Monitoring, and Governance

Calibration diagnostics, monitoring thresholds, and governance files were added to frame the model as decision support. The documentation now includes intended use, prohibited uses, human-review policy, monitoring plan, model-card template, validation-report template, limitations, and a deployment runbook.

The model is not approved for autonomous coverage, benefit, pricing, reserving, clinical, or adverse consumer decisions.

## Backend API

The FastAPI backend exposes `/health`, `/metadata`, `/predict`, `/state`, `/recommend_action`, `/simulate`, and `/decision_support`. Responses include model name, model version, risk score, risk tier, operating action, reason codes, input-review flags, and human-review indicators.

The model serving contract is explicit:

```json
{
  "model_name": "high_cost_claim_classifier",
  "model_version": "actuarial_decision_support_v1",
  "python_version": "3.11.10",
  "sklearn_version": "1.3.0",
  "feature_version": "gold_features_v2_utilization_chronic_structure",
  "split_version": "xxhash64_bene_id_mod_100_v1",
  "target_definition": "next_year_top_decile_training_threshold",
  "artifact_source": "backend/model_artifacts/model/MLmodel",
  "mlflow_run_id": "347f8810d18d447b954eabb5be84f49e"
}
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
- `python3 -m compileall backend tests test_project.py`
- `pytest`
- `./scripts/run_local_tests.sh`
- `python3 test_project.py`
- Backend `/health` and `/metadata` smoke checks.
- Streamlit frontend reachability check.

Current local result: 22 passed, 1 skipped. The skipped test is the artifact reproduction check that only enforces sklearn 1.3.0 when running under the artifact's Python 3.11 line.

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
