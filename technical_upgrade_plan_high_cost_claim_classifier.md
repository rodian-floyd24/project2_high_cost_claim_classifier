# Technical Upgrade Plan for `project2_high_cost_claim_classifier`

## Purpose

This document records the detailed technical upgrades needed to move the high-cost claim classifier from a strong class-project prototype toward a cleaner, more reproducible, portfolio-grade machine learning system.

The project already has a strong foundation:

- Medallion-style Databricks pipeline: bronze, silver, gold.
- Prospective target design using year \(t\) features to predict year \(t+1\) high-cost status.
- Train-only high-cost threshold computation.
- Beneficiary-level hash holdout split.
- Logistic regression, tree-based, gradient boosting, and XGBoost modeling paths.
- PR-AUC, ROC-AUC, Brier score, calibration, top-k capture, and lift diagnostics.
- FastAPI backend.
- Streamlit frontend.
- Simulated MDP/Q-learning decision-support layer.
- Governance, monitoring, validation, and model-card documentation.

The remaining work is mostly about **artifact consistency, reproducibility, calibration proof, serving-contract integrity, and production-style documentation**.

---

## Executive Summary

The most important issue is that the repository currently has a mismatch between:

1. the current shared feature contract,
2. the model training scripts,
3. the served MLflow artifact,
4. the model metadata,
5. the backend dependency pins,
6. the frontend hardcoded performance values.

In a machine learning system, this is not a cosmetic issue. The trained model is a mathematical object defined over a specific feature space:

\[
f: \mathcal{X} \rightarrow [0,1],
\]

where \(\mathcal{X}\) is the ordered set of model inputs. If the repository says the model uses one feature space but the deployed artifact uses another, the system is not fully reproducible.

The highest-priority upgrade is therefore:

> **Make the feature contract, trained artifact, metadata, dependency environment, tests, backend, frontend, and documentation agree.**

---

# Upgrade 1: Fix the Artifact and Dependency Version Contradiction

## Current Problem

The served MLflow artifact and model metadata indicate that the model was created under:

```text
Python: 3.11.10
scikit-learn: 1.3.0
```

However, the backend and serving requirements currently pin:

```text
scikit-learn==1.7.2
```

This is a reproducibility problem. Pickled scikit-learn models are not guaranteed to be stable across major or minor version jumps. The backend currently includes a compatibility shim for old gradient-boosting internals, which is a sign that the model artifact and serving environment are not cleanly aligned.

## Why This Matters

A model artifact is not just a file. It is a serialized estimator whose behavior depends on:

- Python version,
- scikit-learn version,
- NumPy version,
- pandas version,
- MLflow version,
- preprocessing pipeline structure,
- feature order,
- categorical encoding behavior.

If the training and serving environments differ, predictions may still run, but the system becomes fragile. In an actuarial or healthcare-risk context, fragile reproducibility is unacceptable.

## Required Upgrade

Choose exactly one of the following strategies.

---

## Option A: Artifact-Faithful Serving

Use the environment that matches the existing artifact.

### Required changes

Update:

```text
backend/requirements.txt
requirements-serving-py311.txt
```

to use:

```text
scikit-learn==1.3.0
```

Then confirm that the backend runs under Python 3.11.

### Advantages

- Fastest path.
- Preserves the existing artifact.
- Avoids relying on compatibility shims.
- Makes metadata truthful.

### Disadvantages

- Uses an older scikit-learn version.
- May conflict with newer development environments.
- May require a Python 3.11-specific serving container.

---

## Option B: Modernized Artifact

Retrain and relog the final model under the newer serving environment.

### Required changes

1. Train the selected final model under:

```text
Python: 3.11.x
scikit-learn: 1.7.2
```

2. Regenerate the MLflow model artifact.

3. Replace:

```text
backend/model_artifacts/model/
backend/model_artifacts/model_metadata.json
```

4. Remove the compatibility shim if no longer needed.

5. Confirm the loaded model works without monkey-patching sklearn internals.

### Advantages

- Cleaner long-term engineering.
- Better compatibility with current libraries.
- Removes fragile compatibility code.
- Better for deployment and portfolio review.

### Disadvantages

- Requires a full retraining and artifact export.
- Metrics may change slightly.
- Documentation and frontend metrics must be regenerated.

---

## Acceptance Criteria

This upgrade is complete when:

- `backend/model_artifacts/model/MLmodel` and `model_metadata.json` report the same sklearn version.
- `backend/requirements.txt` and `requirements-serving-py311.txt` match the artifact version.
- The backend loads the model without warnings about sklearn incompatibility.
- The compatibility shim is either removed or explicitly justified in documentation.
- `/health` reports the same version as the artifact metadata.
- Tests verify dependency-artifact consistency.

---

# Upgrade 2: Regenerate the Served Artifact from the Current Feature Contract

## Current Problem

The repository defines a large v2 feature contract in:

```text
shared/feature_contract.py
```

This includes many engineered features:

- enrollment features,
- chronic burden features,
- chronic-condition flags,
- age interactions,
- utilization interactions,
- cost log transforms,
- prior-year features,
- cost trend features,
- high-cost history features,
- cost-share features,
- categorical interaction bands.

However, the currently served MLflow artifact appears to use a much smaller feature signature. This means the deployed model may not represent the latest training design.

## Why This Matters

The model input space must be explicitly defined and stable:

\[
\mathbf{x}
=
(x_1, x_2, \ldots, x_p).
\]

If the training script uses one vector \(\mathbf{x}\), while the served artifact expects a different vector \(\tilde{\mathbf{x}}\), then the repository is no longer internally coherent.

Even if the app still runs, the system becomes difficult to defend because the documentation, training code, and serving code are not describing the same estimator.

## Required Upgrade

Regenerate the deployed artifact from the current final training pipeline.

### Steps

1. Run the final Databricks training workflow using the current feature contract.
2. Confirm that the selected final model uses:

```python
MODEL_FEATURE_ORDER
```

from:

```text
shared/feature_contract.py
```

3. Export the final model artifact.
4. Replace the backend artifact directory:

```text
backend/model_artifacts/model/
```

5. Regenerate:

```text
backend/model_artifacts/model_metadata.json
```

6. Confirm that the model signature matches the intended feature order.

## Add a Feature-Contract Test

Create a test such as:

```python
def test_model_signature_matches_feature_contract():
    model = mlflow.sklearn.load_model("backend/model_artifacts/model")
    signature_features = [item.name for item in model.metadata.signature.inputs.inputs]

    assert signature_features == MODEL_FEATURE_ORDER
```

If the deployed model intentionally uses a reduced feature set, define a separate explicit artifact contract:

```python
SERVED_MODEL_FEATURE_ORDER = [...]
```

and test against that.

Do not allow an undocumented mismatch.

## Acceptance Criteria

This upgrade is complete when:

- The served model signature is intentionally defined.
- The artifact feature order matches either `MODEL_FEATURE_ORDER` or an explicitly documented served-feature contract.
- `model_metadata.json` contains the exact feature version and run ID.
- Backend scoring uses the same feature order as the artifact.
- The README no longer implies that the served model uses a feature set different from the actual artifact.

---

# Upgrade 3: Make Calibration Verifiable

## Current Problem

The project claims that the deployed model is calibrated, and the metadata lists:

```json
"calibration_method": "isotonic_regression",
"calibration_status": "calibrated"
```

The repository also contains calibration diagnostics, including:

- Brier score,
- mean predicted probability,
- observed positive rate,
- absolute calibration gap,
- decile calibration,
- reliability curve points.

However, the loaded artifact should be programmatically verified as calibrated. Documentation alone is not enough.

## Why This Matters

For classification models, discrimination and calibration are different properties.

A model can rank patients well while still producing poor probabilities. Ranking quality is measured by metrics such as:

\[
\text{ROC-AUC}, \quad \text{PR-AUC}, \quad \text{top-k capture}, \quad \text{lift}.
\]

Probability quality is measured by:

\[
\text{Brier Score}
=
\frac{1}{n}\sum_{i=1}^{n}(\hat{p}_i - y_i)^2,
\]

and by calibration diagnostics comparing predicted probabilities to observed frequencies.

For this project, calibration matters because the app displays output as a risk probability, not only as a rank score.

## Required Upgrade

Add a test that confirms the artifact is actually calibrated.

Possible checks:

```python
from sklearn.calibration import CalibratedClassifierCV

def test_served_model_is_calibrated():
    model = mlflow.sklearn.load_model("backend/model_artifacts/model")

    assert isinstance(model, CalibratedClassifierCV) or "calibrat" in type(model).__name__.lower()
```

If the model is a pipeline:

```python
def test_pipeline_contains_calibration_step():
    model = mlflow.sklearn.load_model("backend/model_artifacts/model")

    step_names = getattr(model, "named_steps", {}).keys()
    assert any("calibr" in step for step in step_names)
```

If calibration is applied outside the sklearn object, then store the calibration method and parameters explicitly in artifact metadata and test for those.

## Add Calibration Gates

Add threshold-based gates such as:

```python
MAX_ABSOLUTE_CALIBRATION_GAP = 0.03
MAX_BRIER_SCORE = 0.08
```

Then fail validation if the selected model exceeds the allowed tolerance.

## Acceptance Criteria

This upgrade is complete when:

- The calibration method is visible in the model artifact or documented artifact metadata.
- A test verifies that the artifact calibration status is truthful.
- The validation report includes Brier score, observed positive rate, mean predicted probability, and calibration gap.
- The frontend probability language is consistent with the calibration quality.
- If calibration is weak, the app says “risk score” instead of overclaiming “probability.”

---

# Upgrade 4: Eliminate Training-Serving Skew

## Current Problem

The Databricks gold pipeline computes many features from population-level or year-level context, such as:

- annual cost percentile,
- annual cost decile,
- annual cost relative to yearly median,
- current-year high-cost indicator,
- prior-year high-cost indicator,
- prior-year trend features.

The backend Streamlit app reconstructs features from user-entered profile values. Some values are either defaulted, approximated, or not visible to the user.

This creates training-serving skew.

## Why This Matters

Training-serving skew occurs when the feature-generating process during training differs from the feature-generating process during inference.

Let:

\[
\phi_{train}(z)
\]

be the training feature transformation and

\[
\phi_{serve}(z)
\]

be the serving feature transformation.

The model assumes:

\[
\phi_{train}(z) = \phi_{serve}(z).
\]

If this equality fails, then predictions are not fully reliable.

## Required Upgrade

Classify every model feature into one of three categories.

---

## Category 1: Direct User Input

Examples:

- age band,
- sex,
- state,
- race code,
- enrollment months,
- chronic condition count,
- claim counts,
- cost amounts,
- provider count.

These are acceptable for direct app input.

---

## Category 2: Deterministic Derived Feature

Examples:

- total claim count,
- log costs,
- cost shares,
- claim rates per enrollment month,
- binary utilization flags,
- chronic burden bands,
- age interactions.

These are acceptable if backend formulas exactly match the Databricks formulas.

---

## Category 3: Population-Relative or Historical Context Feature

Examples:

- annual cost percentile,
- annual cost decile,
- annual cost-to-year median,
- current-year high-cost indicator,
- prior-year high-cost indicator,
- high-cost last two years,
- trend ratio.

These require careful handling.

## Required Decision

For Category 3 features, choose one approach:

### Approach A: Remove from deployed model

Train a deployment-safe model that only uses features available at inference time.

### Approach B: Require explicit inputs

Expose the fields in the API and frontend, and validate them.

### Approach C: Compute from a reference distribution

Store yearly reference statistics and compute percentile/decile/median-relative values during serving.

## Acceptance Criteria

This upgrade is complete when:

- Every served model feature is labeled as direct, derived, or context-dependent.
- Backend formulas match Databricks formulas.
- No hidden default values materially affect predictions without documentation.
- The app explains which values are simulated.
- Tests compare backend feature engineering against a known Databricks-style fixture.

---

# Upgrade 5: Replace Hardcoded Frontend Metrics with Artifact Metrics

## Current Problem

The Streamlit frontend hardcodes project metrics such as:

```python
PROJECT_METRICS = [
    ("Test PR-AUC", "0.465", ...),
    ("Top-5% Capture", "28.8%", ...),
    ("Top-10% Capture", "43.0%", ...),
    ("Brier Score", "0.073", ...)
]
```

Hardcoded metrics can become stale whenever the model is retrained.

## Why This Matters

The frontend is part of the model communication layer. If the displayed metrics do not match the deployed model, then the app becomes misleading.

## Required Upgrade

Create:

```text
backend/model_artifacts/model_metrics.json
```

Example:

```json
{
  "model_name": "gradient_boosting",
  "run_id": "...",
  "split_name": "test",
  "positive_rate": 0.0447,
  "roc_auc": 0.8074,
  "pr_auc": 0.3755,
  "brier_score": 0.0487,
  "top_5_capture_rate": 0.2437,
  "top_5_lift": 4.265,
  "top_10_capture_rate": 0.3839,
  "top_10_lift": 3.839,
  "calibration_method": "isotonic_regression",
  "processed_at_utc": "..."
}
```

Then expose those metrics through:

```text
GET /metadata
```

or a new endpoint:

```text
GET /model_metrics
```

The frontend should fetch those values dynamically.

## Acceptance Criteria

This upgrade is complete when:

- Streamlit does not hardcode final performance metrics.
- Displayed metrics come from the same artifact as the deployed model.
- The metric artifact includes run ID, split name, split version, feature version, and timestamp.
- The README and frontend report the same final values.

---

# Upgrade 6: Strengthen MLflow Logging

## Current Problem

Some training scripts still use the older positional MLflow logging pattern:

```python
mlflow.sklearn.log_model(model, "model", signature=signature)
```

Modern MLflow prefers explicit named arguments.

## Required Upgrade

Update model logging to use:

```python
mlflow.sklearn.log_model(
    sk_model=model,
    name="model",
    signature=signature,
    input_example=input_example,
)
```

If the installed MLflow version does not support `name`, then explicitly pin the MLflow version and document the reason for using `artifact_path`.

## Add Realistic Input Examples

The input example should include:

- missing values where realistic,
- float-safe casting for nullable integer-like columns,
- categorical levels present in training,
- representative values from the processed training data.

## Acceptance Criteria

This upgrade is complete when:

- MLflow logging produces no deprecation warning.
- The logged model contains a realistic input example.
- The logged signature matches the final feature contract.
- Nullable integer warnings are resolved or explicitly documented.

---

# Upgrade 7: Add a Complete Artifact Consistency Test Suite

## Current Problem

The repository has useful tests, especially for leakage and metric utilities. However, it needs stronger tests tying together:

- feature contract,
- model artifact,
- metadata,
- dependency pins,
- backend scoring,
- frontend metrics.

## Required Tests

Add a file:

```text
tests/test_artifact_consistency.py
```

Suggested tests:

```python
def test_model_metadata_matches_mlmodel():
    metadata = json.load(open("backend/model_artifacts/model_metadata.json"))
    mlmodel = yaml.safe_load(open("backend/model_artifacts/model/MLmodel"))

    assert metadata["sklearn_version"] == str(mlmodel["flavors"]["sklearn"]["sklearn_version"])
    assert metadata["python_version"] == str(mlmodel["flavors"]["python_function"]["python_version"])
```

```python
def test_backend_requirements_match_artifact_sklearn_version():
    metadata = json.load(open("backend/model_artifacts/model_metadata.json"))
    requirements = Path("backend/requirements.txt").read_text()

    assert f"scikit-learn=={metadata['sklearn_version']}" in requirements
```

```python
def test_model_signature_has_no_target_columns():
    model = mlflow.sklearn.load_model("backend/model_artifacts/model")
    signature_features = [item.name for item in model.metadata.signature.inputs.inputs]

    forbidden_prefixes = ("target_", "next_year_")
    forbidden_columns = {"label", "target_annual_claim_cost"}

    for feature in signature_features:
        assert feature not in forbidden_columns
        assert not feature.startswith(forbidden_prefixes)
```

```python
def test_model_signature_is_documented():
    model = mlflow.sklearn.load_model("backend/model_artifacts/model")
    signature_features = [item.name for item in model.metadata.signature.inputs.inputs]

    assert len(signature_features) > 0
```

## Acceptance Criteria

This upgrade is complete when:

- Artifact metadata and MLmodel agree.
- Dependency pins agree with artifact metadata.
- No target columns appear in the artifact signature.
- Feature contract mismatch is impossible to miss.
- CI fails if a stale model artifact is committed.

---

# Upgrade 8: Add Backend Feature-Engineering Equivalence Tests

## Current Problem

The backend manually reconstructs engineered features from a profile. The Databricks gold pipeline also constructs engineered features. These two formulas must remain equivalent.

## Why This Matters

If the backend computes:

\[
\text{claims\_per\_enrollment\_month}
=
\frac{\text{claims}}{\text{months}},
\]

but Databricks uses a slightly different denominator rule, missing-value rule, or clipping rule, then predictions can drift.

## Required Upgrade

Create a small fixture:

```text
tests/fixtures/beneficiary_profile_example.json
tests/fixtures/expected_engineered_features.json
```

Then test:

```python
def test_backend_feature_engineering_matches_expected_fixture():
    profile = BeneficiaryProfile(**json.load(open("tests/fixtures/beneficiary_profile_example.json")))
    features = build_feature_row(profile)
    expected = json.load(open("tests/fixtures/expected_engineered_features.json"))

    for key, expected_value in expected.items():
        assert features[key] == expected_value
```

For floating-point values:

```python
assert abs(features[key] - expected_value) < 1e-9
```

## Acceptance Criteria

This upgrade is complete when:

- Backend feature engineering has deterministic tests.
- Cost shares sum to approximately 1 when total cost is positive.
- Safe-rate behavior is tested for zero enrollment months.
- Log-transform behavior is tested for zero and positive costs.
- Chronic burden bands are tested at boundary values 0, 2, 5, and 6.

---

# Upgrade 9: Separate Portfolio Demo Mode from Production-Style Scoring Mode

## Current Problem

The frontend behaves like a scenario simulator, while the backend and documentation sometimes describe it as a deployed risk scoring system.

Those are not exactly the same thing.

## Required Upgrade

Explicitly define two modes:

---

## Mode 1: Demo / Scenario Mode

Purpose:

- Let users manipulate a hypothetical beneficiary profile.
- Show how utilization, chronic burden, and costs affect risk.
- Demonstrate the API and decision-support workflow.

Required language:

```text
This is a scenario simulator using user-entered values. It is not a live actuarial production scoring system.
```

---

## Mode 2: Batch / Production-Style Scoring Mode

Purpose:

- Score real beneficiary-year rows from the gold table.
- Use exact feature transformations.
- Preserve feature contract and artifact consistency.

Required input:

```text
gold_beneficiary_year_features
```

or an equivalent validated feature table.

## Acceptance Criteria

This upgrade is complete when:

- Streamlit clearly labels scenario mode.
- The backend has a documented production-style scoring path.
- The README distinguishes scenario simulation from batch scoring.
- The deployed app does not imply that manually entered profiles are equivalent to full production features.

---

# Upgrade 10: Improve Model Selection Documentation

## Current Problem

The repo includes multiple models and comparison scripts, but the final model-selection logic should be stated more explicitly.

## Required Upgrade

Add a document:

```text
docs/model_selection_rationale.md
```

Include:

1. Candidate models.
2. Training split.
3. Validation split.
4. Test split.
5. Target definition.
6. Feature version.
7. Metric hierarchy.
8. Final selected model.
9. Reason selected.
10. Known limitations.

## Recommended Metric Hierarchy

For this problem, accuracy should not be primary because the event is imbalanced.

Recommended ordering:

1. PR-AUC.
2. Top-5% capture.
3. Top-10% capture.
4. Calibration gap.
5. Brier score.
6. ROC-AUC.
7. Precision, recall, and F1 at the selected operating threshold.
8. Accuracy as a secondary descriptive metric only.

## Acceptance Criteria

This upgrade is complete when:

- The final model selection is reproducible from logged metrics.
- The final model is not chosen using test-set tuning.
- The validation split is used for threshold and calibration decisions.
- The test split is used only for final reporting.
- The README summarizes the model-selection rationale.

---

# Upgrade 11: Improve Calibration Language in the App

## Current Problem

The app displays values as probabilities. That is acceptable only if calibration is strong and verified.

## Required Upgrade

Use language conditional on calibration quality.

If calibration is verified:

```text
Calibrated next-year high-cost probability
```

If calibration is weak or unverified:

```text
Risk score
```

or:

```text
Estimated risk score used for ranking and tiering
```

## Acceptance Criteria

This upgrade is complete when:

- The frontend does not overclaim probability quality.
- The displayed label is controlled by model metadata.
- The metadata includes calibration status and calibration gap.
- The app explains that predictions are decision-support estimates, not deterministic forecasts.

---

# Upgrade 12: Add a Model Card with Actual Final Numbers

## Current Problem

The repository has governance and validation documentation, but the final deployed model should have a concrete model card.

## Required Upgrade

Create:

```text
docs/final_model_card.md
```

Include:

## Required Sections

1. Model name.
2. Model version.
3. Artifact run ID.
4. Dataset source.
5. Target definition.
6. Feature timing.
7. Split strategy.
8. Feature version.
9. Training algorithm.
10. Calibration method.
11. Evaluation population.
12. Final test metrics.
13. Top-k capture and lift.
14. Calibration diagnostics.
15. Intended use.
16. Prohibited use.
17. Known limitations.
18. Monitoring plan.
19. Retraining triggers.

## Acceptance Criteria

This upgrade is complete when:

- The model card references the exact deployed artifact.
- The metrics match `model_metrics.json`.
- The model card states that the RL layer is simulated.
- The model card distinguishes ranking quality from probability calibration.

---

# Upgrade 13: Add Monitoring Thresholds and Drift Gates

## Current Problem

The repo has monitoring documentation and scripts, but monitoring should be connected to explicit thresholds.

## Required Upgrade

Define monitoring gates such as:

```python
MAX_PREDICTION_MEAN_SHIFT = 0.03
MAX_BRIER_SCORE_DEGRADATION = 0.02
MAX_ABSOLUTE_CALIBRATION_GAP = 0.03
MIN_TOP_10_CAPTURE_RATE = 0.30
MAX_FEATURE_MISSING_RATE = 0.05
```

Track:

- prediction mean,
- observed positive rate,
- calibration gap,
- top-k capture,
- feature missingness,
- feature distribution drift,
- categorical level drift,
- score distribution drift.

## Acceptance Criteria

This upgrade is complete when:

- Monitoring thresholds are explicit.
- Monitoring outputs pass/fail statuses.
- Drift checks are linked to retraining recommendations.
- Monitoring results are included in the validation packet.

---

# Upgrade 14: Improve README Structure

## Current Problem

The README is strong, but it can be made more recruiter- and professor-friendly.

## Required Upgrade

Use the following structure:

```text
# High-Cost Claim Classifier

## One-Sentence Summary
## Problem Statement
## Dataset and Target Definition
## Pipeline Architecture
## Feature Engineering
## Modeling Approach
## Evaluation Metrics
## Final Results
## Calibration and Ranking Diagnostics
## API and Frontend
## RL Decision-Support Prototype
## Governance and Limitations
## Reproducibility
## Repository Structure
## How to Run
## Future Work
```

## Key Language to Include

```text
The supervised model estimates next-year high-cost risk using current-year beneficiary, utilization, chronic burden, and cost features.
```

```text
The target is defined using a training-only top-decile threshold to avoid leakage from validation and test rows.
```

```text
Because the positive class is imbalanced, PR-AUC, top-k capture, lift, and calibration diagnostics are emphasized over raw accuracy.
```

```text
The RL layer is a simulated policy prototype and is not interpreted as causal evidence of intervention effectiveness.
```

## Acceptance Criteria

This upgrade is complete when:

- The README clearly separates model, app, and policy prototype.
- The final metrics are pulled from the final model artifact.
- The README does not overclaim clinical, actuarial, or causal validity.
- The run instructions match the actual repo layout.

---

# Upgrade 15: Add a Reproducibility Packet

## Required Files

Create a folder:

```text
validation_packet/
```

Include:

```text
validation_packet/
  final_model_card.md
  final_metrics.json
  calibration_summary.csv
  calibration_deciles.csv
  topk_capture_lift.csv
  feature_contract.json
  model_signature.json
  artifact_environment.json
  leakage_check_results.txt
  test_results.txt
```

## Purpose

This gives professors, recruiters, and technical reviewers one place to verify the system.

## Acceptance Criteria

This upgrade is complete when:

- The packet can be regenerated from scripts.
- The packet includes the final run ID.
- The packet includes the final split version.
- The packet includes final model metrics.
- The packet includes artifact environment details.

---

# Upgrade 16: Clean Up Naming and Versioning

## Current Problem

There are several similar concepts:

- model version,
- feature version,
- split version,
- target version,
- artifact version,
- app version,
- RL policy version.

These should be explicit and consistent.

## Required Upgrade

Define a single metadata schema:

```json
{
  "model_name": "...",
  "model_version": "...",
  "artifact_version": "...",
  "feature_version": "...",
  "target_definition_version": "...",
  "split_version": "...",
  "calibration_version": "...",
  "policy_layer_version": "...",
  "training_run_id": "...",
  "databricks_refresh_run_id": "...",
  "created_at_utc": "..."
}
```

## Acceptance Criteria

This upgrade is complete when:

- Every major component has a version.
- `/metadata` exposes all versions.
- README and model card use the same versions.
- Tests verify metadata completeness.

---

# Upgrade 17: Add Clear Final Limitations

## Required Limitation Statements

Include these in README, model card, and app:

```text
This model is trained on synthetic CMS-style claims data and should not be interpreted as validated performance on real Medicare claims.
```

```text
The supervised model estimates risk using observed historical patterns, not causal effects.
```

```text
The RL decision-support layer is a simulated policy prototype and should not be interpreted as evidence that an intervention will reduce cost.
```

```text
The tool is not intended for autonomous coverage, pricing, reserving, underwriting, clinical, or adverse consumer decisions.
```

```text
Predicted probabilities require calibration monitoring before operational use.
```

## Acceptance Criteria

This upgrade is complete when:

- Limitations are stated in plain language.
- The app does not overclaim.
- The model card includes intended and prohibited uses.
- The RL layer is clearly labeled as simulated.

---

# Recommended Implementation Order

## Phase 1: Fix Reproducibility

1. Decide sklearn strategy:
   - artifact-faithful serving, or
   - modernized retraining.
2. Make requirements match artifact metadata.
3. Add artifact consistency tests.
4. Confirm backend `/health` works.

## Phase 2: Fix Feature Contract Alignment

1. Regenerate final model artifact.
2. Confirm model signature.
3. Update model metadata.
4. Add model signature test.
5. Remove undocumented feature-contract mismatch.

## Phase 3: Fix Calibration Proof

1. Confirm calibration object or calibration metadata.
2. Add calibration verification tests.
3. Generate calibration diagnostics.
4. Update app language based on calibration status.

## Phase 4: Fix Frontend Metrics

1. Create `model_metrics.json`.
2. Expose metrics through backend.
3. Fetch metrics dynamically in Streamlit.
4. Remove hardcoded metric constants.

## Phase 5: Strengthen Documentation

1. Add final model card.
2. Add model-selection rationale.
3. Add validation packet.
4. Update README.
5. Add limitation language.

---

# Final Target State

After the upgrades, the repository should satisfy the following invariant:

```text
feature_contract
= training_features
= artifact_signature
= backend_scoring_features
= metadata_feature_version
= README/model_card_description
```

The final deployed system should be describable as:

> A reproducible actuarial machine learning pipeline that uses current-year beneficiary, utilization, chronic burden, and cost features to estimate next-year high-cost risk, evaluated on a beneficiary-level held-out test split with ranking, calibration, and top-k capture diagnostics, and served through a documented FastAPI and Streamlit decision-support prototype.

That is the standard needed for an A-level technical portfolio project.
