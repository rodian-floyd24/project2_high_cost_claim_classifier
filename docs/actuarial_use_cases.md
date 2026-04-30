# Actuarial Use Cases

## Intended Use

The model estimates the probability that a beneficiary-year will become high cost in the following calendar year. The model is intended for care-management queue prioritization, actuarial risk segmentation, and experience-study support.

The approved first use case is human-reviewed claims triage and care-management prioritization. The score ranks beneficiaries so limited review capacity can focus on members with higher predicted next-year high-cost risk.

## Prohibited Uses

The model must not be used as the sole basis for:

- denying coverage,
- changing benefits,
- pricing insurance contracts,
- booking reserves,
- making clinical decisions,
- making adverse consumer decisions without human review.

## Materiality

Materiality tier: medium for the project prototype. The model influences a simulated care-management queue, but it is not approved to directly change premiums, reserves, benefits, coverage, or clinical treatment.

## Users And Decision Owners

Primary users are actuaries, analytics reviewers, and care-management operations staff. The decision owner remains a human reviewer. Model output is decision support, not autonomous decision-making.

## Model Inventory Row

| Field | Value |
|---|---|
| model_id | high_cost_medicare_risk_v1 |
| model_name | gradient_boosting_high_cost_risk |
| business_use | care-management queue prioritization |
| intended_use | next-year high-cost risk ranking |
| prohibited_use | autonomous adverse consumer, pricing, reserving, benefit, coverage, or clinical decisions |
| materiality_tier | medium prototype |
| data_sources | CMS DE-SynPUF synthetic Medicare claims |
| model_type | gradient boosting challenger with logistic regression baseline |
| champion_model | gradient_boosting |
| challenger_models | logistic_regression, random_forest, xgboost |
| approval_status | prototype only |
| monitoring_frequency | per refresh and before any promotion |
