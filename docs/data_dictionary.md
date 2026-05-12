# Gold Feature Data Dictionary

Grain: one row per `bene_id` and calendar `year`.

Primary key: `bene_id`, `year`.

Feature version: `gold_features_v2_utilization_chronic_structure`.

Canonical feature contract: `shared/feature_contract.py`.

| Column | Description |
|---|---|
| bene_id | Synthetic beneficiary identifier. |
| year | Feature calendar year. |
| annual_claim_cost | Total observed claim cost in the feature year. |
| enrollment_months_count | Months of beneficiary enrollment in the feature year. |
| chronic_condition_count | Count of chronic condition flags, expected range 0 to 11. |
| chronic_burden_band | Categorical chronic burden band. |
| total_claim_count | Total inpatient, outpatient, carrier, and PDE claim count. |
| claims_per_enrollment_month | Claim count normalized by enrollment months. |
| unique_provider_count | Distinct provider count across claim settings. |
| provider_fragmentation_index | Unique providers divided by total claims. |
| prior_year_annual_claim_cost | Prior feature-year annual cost, if available. |
| current_year_high_cost_indicator | Feature-year top-decile indicator for history only. |
| target_annual_claim_cost | Next-year cost in the prospective modeling frame only. |
| label | Training label derived from the training-only top-decile target threshold. |
