# Technical Upgrade Walkthrough

The High-Cost Claim Classifier has been upgraded into a more reproducible, portfolio-grade machine learning system. The upgrades focus on artifact consistency, feature-contract alignment, backend scoring reliability, dynamic model metrics, live risk-score clarity, and governance documentation.

## 1. Reproducibility Fixes

The serving environment was aligned with the serialized MLflow artifact.

The deployed model artifact was trained under scikit-learn `1.3.0`. The backend serving requirements are now pinned to:

```text
scikit-learn==1.3.0
```

in both:

```text
backend/requirements.txt
requirements-serving-py311.txt
```

The artifact consistency test in `tests/test_artifact_consistency.py` verifies that the serving requirements match the MLflow artifact metadata.

**Result:** the backend can serve the model in the same dependency environment used to create the artifact, while local Python 3.12 development remains covered by compatibility tests and warnings.

## 2. Feature Contract Alignment

The repository now separates the full Databricks modeling feature contract from the exact deployed artifact serving contract.

The full Databricks modeling feature universe remains defined in:

```text
shared/feature_contract.py
```

The currently deployed MLflow artifact expects a smaller 36-feature input signature. That serving-specific contract is explicit:

```python
SERVED_FEATURE_VERSION = "served_artifact_36_features_v1"
SERVED_MODEL_FEATURE_ORDER = [...]
```

The artifact consistency suite verifies:

```python
artifact_signature == SERVED_MODEL_FEATURE_ORDER
```

**Result:** the deployed artifact's feature space is documented and continuously tested instead of being inferred from the broader training feature universe.

## 3. Backend Scoring Equivalence

The backend scoring pipeline now falls back to:

```python
SERVED_MODEL_FEATURE_ORDER
```

instead of the broader:

```python
MODEL_FEATURE_ORDER
```

This prevents the backend from constructing a feature matrix that does not match the artifact signature.

`tests/test_backend_scoring_equivalence.py` verifies deterministic feature calculations including:

```text
cost_per_enrollment_month
claims_per_enrollment_month
provider_fragmentation_index
annual_cost_log1p
inpatient_cost_share
outpatient_cost_share
carrier_cost_share
rx_cost_share
```

**Result:** backend feature engineering is tested against the serving contract, reducing training-serving skew.

## 4. Live Risk Score

The live scoring layer now returns a decision-support response rather than a bare model probability.

The prediction API exposes:

```text
raw_model_probability
calibrated_probability
risk_score_0_100
risk_tier
intervention_flag
decision_threshold
threshold_source
reason_codes
feature_contract_version
calibration_method
split_version
```

The displayed score is a monotone 0-100 percentile-rank score over `backend/model_artifacts/reference_distribution.json`. The raw model probability remains available for diagnostics, but the UI emphasizes the calibrated probability, live risk score, risk tier, intervention flag, and top reason codes.

**Result:** the app behaves more like an actuarial decision-support tool and no longer presents the raw probability as the primary user-facing score.

## 5. Dynamic Artifact Metrics in the Frontend

Model performance metrics are stored in:

```text
backend/model_artifacts/model_metrics.json
```

The backend exposes:

```text
GET /model_metrics
```

The Streamlit frontend dynamically fetches and displays those artifact-level metrics, with fallback messaging if the metrics artifact or backend endpoint is unavailable.

**Result:** the dashboard is tied to the deployed model artifact instead of relying on stale hardcoded values.

## 6. Documentation and Governance

The project documentation now makes the prototype boundary explicit.

Updated files include:

```text
docs/final_model_card.md
backend/monitoring_gates.py
README.md
writeup_source.md
writeup.pdf
```

The model card documents intended use, prohibited use, the serving feature contract, artifact assumptions, calibration expectations, monitoring expectations, and scenario-mode limitations.

Monitoring gates define threshold logic for:

```text
feature drift
capture-rate decay
Brier score tracking
calibration degradation
population positive-rate drift
```

**Result:** the project has clearer governance boundaries and avoids overclaiming production, clinical, or actuarial validity.

## Validation Results

The local validation suite was executed with:

```bash
./scripts/run_local_tests.sh
```

Result:

```text
71 passed, 1 skipped
PASS: no target leakage feature columns or threshold-leakage patterns found in training scripts
```

Focused live-risk-score verification also passed:

```bash
python3 -m pytest -q tests/test_live_risk_score.py tests/test_api_schema.py tests/test_backend_scoring.py tests/test_scoring_consistency.py
```

Result:

```text
20 passed
```

The grading smoke script passes locally and against the deployed URL:

```bash
python3 test_project.py
PROJECT2_API_URL="https://rayodian-ncf.com" python3 test_project.py
```

## Final Technical Assessment

This upgrade closes the largest reproducibility gap in the project.

Before the upgrade, the repository risk was:

```text
training feature contract != served artifact signature != dependency environment
```

After the upgrade, the serving path is cleaner:

```text
served artifact signature = SERVED_MODEL_FEATURE_ORDER = backend scoring contract = tested artifact contract
```

The project is now more defensible as a portfolio-grade machine learning system because it has explicit artifact versioning, an explicit serving feature contract, dependency-artifact consistency, dynamic model metrics, backend feature equivalence tests, monitoring gates, model-card documentation, scenario-mode disclaimers, live risk scoring, deterministic reason codes, and passing test coverage.

## Remaining Future Upgrade

The main future improvement is a Databricks retraining pass that trains a fresh final model using the full `MODEL_FEATURE_ORDER`, then exports a new artifact whose MLflow signature matches the full feature contract.

Until that retraining is available, using `SERVED_MODEL_FEATURE_ORDER` is the correct engineering decision. It is honest, reproducible, and testable.
