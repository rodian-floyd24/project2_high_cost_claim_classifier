# Model Card: High-Cost Claim Classifier

## 1. Model Identification
- **Model name**: high_cost_claim_classifier
- **Version**: actuarial_decision_support_v1
- **Feature Version**: served_artifact_36_features_v1
- **Owner**: Analytics Team
- **Developer**: Analytics Team
- **Validator**: Actuarial Review Board
- **Approval status**: Approved for portfolio demo and development

## 2. Intended Use
Provides next-year high-cost claim probability estimates to support targeted case management and actuarial decision support. The model rank-orders risk to prioritize manual review and simulated interventions for complex beneficiaries.

## 3. Prohibited Use
Must not be used for direct benefit denial, autonomous claim adjudication, pricing individual policies without human review, or clinical diagnostic purposes. It is a decision-support tool, not an automated medical necessity determinator.

## 4. Data
- **Source data**: Synthetic CMS Medicare SynPUF data (claims and profiles)
- **Training period**: 2008-2009 features predicting 2010 outcomes
- **Validation period**: Beneficiary hash-based validation split
- **Test period**: Beneficiary hash-based holdout test set (`xxhash64_bene_id_mod_100_v2_beneficiary_hash_holdout`)
- **Grain**: Beneficiary-year
- **Target definition**: `next_year_top_decile_training_threshold`

## 5. Methodology
- **Model type**: Gradient Boosting Classifier (`scikit-learn==1.3.0`)
- **Features**: 36 claims, utilization, demographic, and chronic condition features (see `SERVED_MODEL_FEATURE_ORDER`). Note: The full gold feature contract defines a broader feature universe, while the served artifact uses a specific 36-feature subset.
- **Preprocessing**: Constant imputation and one-hot encoding
- **Hyperparameters**: Tuned via Databricks AutoML/hyperopt tracking
- **Threshold rule**: `score >= 0.20` for high-cost classification (tuned for F1-score balance)
- **Calibration**: Isotonic regression

## 6. Performance
- **ROC-AUC**: 0.8333
- **PR-AUC**: 0.4653
- **Brier score**: 0.0730
- **Top-5 capture**: 28.85%
- **Top-10 capture**: 43.01%

## 7. Validation
- **Beneficiary-hash holdout validation**: Strict beneficiary separation ensures no data leakage across time for the same patient.
- **Limitations**: The model is calibrated on historical synthetic data and its probabilities represent rank-order usefulness, not a definitive prediction of individual claims.

## 8. Monitoring
- **Frequency**: Monthly scoring drift checks
- **Metrics**: Population risk distribution, top-10 capture rate stability, feature drift
- **Owner**: MLOps Team

## 9. Governance
- **Approval date**: 2026-05-11
- **Next review date**: 2026-11-11
- **Change log**: Migrated to explicit 36-feature contract `SERVED_MODEL_FEATURE_ORDER`. Pinned serving environment to `scikit-learn==1.3.0` to ensure strict artifact reproducibility.
