# Model Card: High-Cost Claim Classifier

## 1. Model Identification
- **Model name**: gradient_boosting
- **Version**: actuarial_decision_support_v2
- **Feature version**: served_artifact_36_features_v1
- **Full feature version**: gold_features_v2_utilization_chronic_structure
- **Split version**: xxhash64_bene_id_mod_100_v2_beneficiary_hash_holdout
- **Owner**: Analytics Team
- **Developer**: Analytics Team
- **Validator**: Actuarial Review Board
- **Approval status**: Approved for portfolio demo and development only

## 2. Intended Use
Provides next-year high-cost claim probability estimates to support targeted case management and actuarial decision support in scenario/demo mode. The model rank-orders risk to prioritize manual review and simulated interventions for complex beneficiaries.

## 3. Prohibited Use
Must not be used for direct benefit denial, autonomous claim adjudication, autonomous coverage decisions, individual pricing, reserving, or clinical diagnostic purposes. It is a decision-support prototype, not an automated medical necessity or adverse-action system.

## 4. Data
- **Source data**: Synthetic CMS Medicare SynPUF data (claims and profiles)
- **Training period**: 2008-2009 features predicting 2010 outcomes
- **Validation period**: Beneficiary hash-based validation split
- **Test period**: Beneficiary hash-based holdout test set (`xxhash64_bene_id_mod_100_v2_beneficiary_hash_holdout`)
- **Grain**: Beneficiary-year
- **Target definition**: `next_year_top_decile_training_threshold`

## 5. Methodology
- **Model type**: Gradient Boosting Classifier (`scikit-learn==1.3.0`)
- **Features**: 36 claims, utilization, demographic, and chronic-condition features (see `SERVED_MODEL_FEATURE_ORDER`). The full Databricks gold feature contract defines a broader feature universe, while the served artifact uses a specific 36-feature subset.
- **Preprocessing**: Constant imputation and one-hot encoding
- **Hyperparameters**: Tuned via Databricks AutoML/hyperopt tracking
- **Decision threshold**: `calibrated_probability >= 0.041` for intervention flagging, sourced from validation F1 threshold tuning
- **Live risk score**: 0-100 percentile rank over the local calibrated-probability reference distribution
- **Risk tiers**: `>=95 very_high`, `>=90 high`, `>=75 elevated`, otherwise `low`
- **Calibration**: Isotonic regression

## 6. Performance
- **ROC-AUC**: 0.8333
- **PR-AUC**: 0.4653
- **Brier score**: 0.0730
- **Top-5 capture**: 28.85%
- **Top-10 capture**: 43.01%

## 7. Validation
- **Beneficiary-hash holdout validation**: Strict beneficiary separation ensures no data leakage across time for the same patient.
- **Artifact consistency**: `tests/test_artifact_consistency.py` verifies the MLflow artifact metadata, serving requirements, and deployed signature.
- **Serving contract**: `tests/test_backend_scoring_equivalence.py` verifies deterministic backend feature calculations and confirms `build_model_frame()` emits `SERVED_MODEL_FEATURE_ORDER`.
- **Limitations**: The model is calibrated on historical synthetic data and its probabilities represent rank-order usefulness, not a definitive prediction of individual claims.

## 8. Monitoring
- **Frequency**: Monthly scoring drift checks
- **Metrics**: Population risk distribution, top-10 capture rate stability, Brier score, calibration degradation, feature drift
- **Owner**: MLOps Team

## 9. Governance
- **Approval date**: 2026-05-12
- **Next review date**: 2026-11-12
- **Change log**: Migrated to explicit 36-feature serving contract `SERVED_MODEL_FEATURE_ORDER`. Pinned serving environment to `scikit-learn==1.3.0` to ensure strict artifact reproducibility. Added live risk score response fields, deterministic reason codes, percentile-rank scoring, and scenario-mode disclaimers.
