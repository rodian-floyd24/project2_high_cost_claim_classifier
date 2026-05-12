# Model Card: High-Cost Claim Classifier

## 1. Model Identification
- **Model name**: gradient_boosting
- **Version**: actuarial_decision_support_v2
- **Feature version**: gold_features_v2_utilization_chronic_structure
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
- **Train-only high-cost threshold**: `10420`

## 5. Methodology
- **Model type**: Gradient Boosting Classifier (`scikit-learn==1.3.0`)
- **Features**: 125 claims, utilization, demographic, chronic-condition, trend, and interaction features. The served artifact signature matches `MODEL_FEATURE_ORDER`.
- **Preprocessing**: Constant imputation and one-hot encoding
- **Hyperparameters**: Tuned via Databricks AutoML/hyperopt tracking
- **Decision threshold**: `calibrated_probability >= 0.20` for intervention flagging, sourced from validation F1 threshold tuning
- **Live risk score**: 0-100 percentile rank over the local calibrated-probability reference distribution
- **Risk tiers**: `>=95 very_high`, `>=90 high`, `>=75 elevated`, otherwise `low`
- **Calibration**: Isotonic regression
- **Cross-validation approach**: The final comparison uses a beneficiary-grouped train/validation/test split rather than row-level k-fold cross-validation to avoid placing the same beneficiary in both training and validation folds. The validation split is used for threshold tuning and model-selection tie-breaking; the held-out test split is reserved for final reporting.

## 6. Performance
- **ROC-AUC**: 0.8333
- **PR-AUC**: 0.4653
- **Brier score**: 0.0730
- **Top-5 capture**: 28.85%
- **Top-10 capture**: 43.01%

## 7. Validation
- **Beneficiary-hash holdout validation**: Strict beneficiary separation ensures no data leakage across time for the same patient.
- **Artifact consistency**: `tests/test_artifact_consistency.py` verifies the MLflow artifact metadata, serving requirements, and deployed signature.
- **Serving contract**: `tests/test_backend_scoring_equivalence.py` verifies deterministic backend feature calculations and confirms `build_model_frame()` emits `MODEL_FEATURE_ORDER`.
- **Limitations**: The model is calibrated on historical synthetic data and its probabilities represent rank-order usefulness, not a definitive prediction of individual claims.

## 8. Monitoring
- **Frequency**: Monthly scoring drift checks
- **Metrics**: Population risk distribution, top-10 capture rate stability, Brier score, calibration degradation, feature drift
- **Owner**: MLOps Team

## 9. Governance
- **Approval date**: 2026-05-12
- **Next review date**: 2026-11-12
- **Change log**: Migrated serving to a Databricks MLflow artifact whose signature matches the full `MODEL_FEATURE_ORDER`. Pinned serving environment to `scikit-learn==1.3.0` to ensure strict artifact reproducibility. Added live risk score response fields, deterministic reason codes, percentile-rank scoring, and scenario-mode disclaimers.
