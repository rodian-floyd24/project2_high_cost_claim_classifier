# Gold Feature EDA Summary

Source: `report_artifacts/project_clean_data_browse.csv`

## Scope

This EDA treats the gold features table as a beneficiary-year modeling table. It checks whether the table is suitable for prospective modeling by building a year `t` to year `t + 1` frame and defining `label = 1` when next-year annual claim cost is above that target year's 90th percentile.

## Table Integrity

| Check | Value |
|---|---:|
| Gold rows | 343,644 |
| Distinct beneficiaries | 116,352 |
| Years | 2008, 2009, 2010 |
| Duplicate `bene_id + year` rows | 0 |
| Rows with missing key fields | 0 |
| Rows with any missing value | 0 |
| Rows with nonpositive enrollment months | 18,854 |
| Rows with a negative cost component | 227 |

## Prospective Modeling Frame

| Check | Value |
|---|---:|
| Year `t` to `t + 1` modeling rows | 227,292 |
| Rows unavailable for prospective target | 116,352 |
| Distinct modeled beneficiaries | 114,538 |
| Feature years | 2008, 2009 |
| Target years | 2009, 2010 |

Class balance by target year:

|   target_year |   row_count |   positive_count | positive_rate   | high_cost_threshold   |   median_target_cost |   mean_target_cost |
|--------------:|------------:|-----------------:|:----------------|:----------------------|---------------------:|-------------------:|
|          2009 |      114538 |            11446 | 9.99%           | $13,300.00            |                 2690 |            5526.96 |
|          2010 |      112754 |            11264 | 9.99%           | $7,290.00             |                 1640 |            3312.45 |

## Chronic Burden Signal

The corrected gold export contains `chronic_condition_count` values from 0 through 11; it is not constant zero. Median annual cost rises monotonically and accelerates as chronic burden increases, which is risk compounding rather than a purely additive linear effect.

|   chronic_condition_count |   row_count | median_annual_claim_cost   | mean_annual_claim_cost   | p90_annual_claim_cost   |
|--------------------------:|------------:|:---------------------------|:-------------------------|:------------------------|
|                         0 |      125983 | $410.00                    | $917.50                  | $2,460.00               |
|                         1 |       44306 | $1,610.00                  | $2,459.90                | $4,830.00               |
|                         2 |       41353 | $2,370.00                  | $3,599.08                | $6,930.00               |
|                         3 |       36871 | $3,240.00                  | $4,938.49                | $9,780.00               |
|                         4 |       31253 | $4,410.00                  | $6,750.15                | $13,768.00              |
|                         5 |       24639 | $5,870.00                  | $8,970.04                | $18,720.00              |
|                         6 |       17785 | $7,910.00                  | $12,101.73               | $25,562.00              |
|                         7 |       11688 | $10,390.00                 | $15,695.33               | $34,009.00              |
|                         8 |        6366 | $13,800.00                 | $20,018.96               | $43,305.00              |
|                         9 |        2613 | $17,270.00                 | $24,581.75               | $53,010.00              |
|                        10 |         701 | $24,390.00                 | $30,882.97               | $60,840.00              |
|                        11 |          86 | $34,795.00                 | $38,957.79               | $70,770.00              |

Chronic-condition row distribution:

|   chronic_condition_count |   row_count | row_rate   |
|--------------------------:|------------:|:-----------|
|                         0 |      125983 | 36.66%     |
|                         1 |       44306 | 12.89%     |
|                         2 |       41353 | 12.03%     |
|                         3 |       36871 | 10.73%     |
|                         4 |       31253 | 9.09%      |
|                         5 |       24639 | 7.17%      |
|                         6 |       17785 | 5.18%      |
|                         7 |       11688 | 3.40%      |
|                         8 |        6366 | 1.85%      |
|                         9 |        2613 | 0.76%      |
|                        10 |         701 | 0.20%      |
|                        11 |          86 | 0.03%      |

Modeling implication: do not rely on a simple linear chronic-count effect alone. The tree models can learn the nonlinear shape directly. Linear baselines should include nonlinear terms, burden bands, or interactions such as the existing age/chronic and chronic-burden-band features.

## Cost and Utilization Shape

Annual cost is highly right-skewed, which supports log cost features and ranking metrics such as PR-AUC, top-k capture, and lift.

| Metric | Value |
|---|---:|
| Annual cost median | $2,040.00 |
| Annual cost 90th percentile | $10,990.00 |
| Annual cost 99th percentile | $44,050.00 |
| Annual cost max | $175,970.00 |

Key cost/count quality checks:

| feature                | zero_rate   |   negative_count |   min |   median |   p90 |   p99 |    max |
|:-----------------------|:------------|-----------------:|------:|---------:|------:|------:|-------:|
| outpatient_total_cost  | 46.21%      |              197 |  -100 |       50 |  1730 |  6170 |  50020 |
| inpatient_total_cost   | 86.76%      |               30 | -3000 |        0 |  5000 | 36000 | 164000 |
| annual_claim_cost      | 12.07%      |                6 | -1330 |     2040 | 10990 | 44050 | 175970 |
| inpatient_claim_count  | 86.46%      |                0 |     0 |        0 |     1 |     3 |     11 |
| outpatient_claim_count | 45.73%      |                0 |     0 |        1 |     7 |    14 |     33 |
| total_claim_days       | 43.41%      |                0 |     0 |        1 |    20 |    65 |    419 |
| rx_total_cost          | 29.14%      |                0 |     0 |      310 |  3020 |  6510 |  14100 |
| pde_claim_count        | 28.46%      |                0 |     0 |        7 |    44 |    78 |    144 |
| carrier_total_cost     | 24.04%      |                0 |     0 |      720 |  3160 |  6570 |  23280 |
| carrier_claim_count    | 23.54%      |                0 |     0 |       10 |    34 |    58 |    171 |
| unique_provider_count  | 21.92%      |                0 |     0 |       14 |    40 |    68 |    162 |
| total_claim_count      | 12.00%      |                0 |     0 |       26 |    71 |   125 |    230 |

## Signal Checks

Top numeric correlations with the next-year high-cost label:

| feature                                       |   correlation_with_next_year_label |
|:----------------------------------------------|-----------------------------------:|
| unique_provider_count                         |                           0.336555 |
| claims_per_month_chronic_count_interaction    |                           0.332243 |
| carrier_claim_count                           |                           0.328572 |
| carrier_total_cost                            |                           0.324697 |
| providers_per_month_chronic_count_interaction |                           0.321503 |
| total_claim_count                             |                           0.314772 |
| claims_per_enrollment_month                   |                           0.309544 |
| chronic_condition_count                       |                           0.293782 |
| total_claim_days                              |                           0.281661 |
| chronic_condition_count_squared               |                           0.280522 |
| carrier_claim_count_log1p                     |                           0.273675 |
| unique_provider_count_log1p                   |                           0.269894 |
| outpatient_claim_count                        |                           0.26679  |
| outpatient_total_cost                         |                           0.262082 |
| outpatient_claim_count_log1p                  |                           0.257535 |

The leading signals are utilization intensity features: unique provider count, carrier claim count, total claim count, claims per enrollment month, and chronic condition count. This is consistent with an actuarial prospective-risk framing: future high-cost status is driven more by stable prior utilization intensity and care engagement patterns than by demographics alone.

Cost-derived feature correlations with the next-year label:

| feature                   |   correlation_with_next_year_label |
|:--------------------------|-----------------------------------:|
| carrier_total_cost        |                           0.324697 |
| outpatient_total_cost     |                           0.262082 |
| annual_claim_cost         |                           0.250885 |
| cost_per_enrollment_month |                           0.23552  |
| rx_total_cost             |                           0.185146 |
| inpatient_total_cost      |                           0.136437 |

Notably, prior-year `annual_claim_cost` is predictive but not the strongest signal. Raw cost is noisy; frequency, provider breadth, and utilization density appear more stable for next-year high-cost risk.

New engineered utilization/chronic structure feature correlations:

| feature                                       |   correlation_with_next_year_label |
|:----------------------------------------------|-----------------------------------:|
| claims_per_month_chronic_count_interaction    |                           0.332243 |
| providers_per_month_chronic_count_interaction |                           0.321503 |
| chronic_condition_count_squared               |                           0.280522 |
| carrier_claim_count_log1p                     |                           0.273675 |
| unique_provider_count_log1p                   |                           0.269894 |
| outpatient_claim_count_log1p                  |                           0.257535 |
| total_claim_count_log1p                       |                           0.228843 |
| inpatient_claim_count_log1p                   |                           0.16536  |
| pde_claim_count_log1p                         |                           0.123092 |

## Pipeline Integrity Notes

The earlier chronic-flag issue was real: the CMS chronic flags use `1/2` coding, and an older parser that expected only `Y/N` would collapse chronic counts to zero. This EDA export is from the corrected gold data and verifies that chronic burden is populated.

Local evidence checked here:

- `report_artifacts/project_clean_data_browse.csv` has `chronic_condition_count` values from 0 to 11.
- The EDA generator derives the new v2 structural features from the export when the CSV predates the Databricks gold refresh.
- The deployed MLflow model signature in `backend/model_artifacts/model/MLmodel` requires `chronic_condition_count`.
- The Databricks training scripts use `default.gold_beneficiary_year_features` and include `chronic_condition_count` plus chronic interactions in their feature lists.
- `databricks/13_gold_pipeline_consistency_check.py` now enforces the live Databricks contract before model results are trusted.

Remaining integrity check before relying on any specific historical model run: confirm that the model artifact being evaluated was trained after the silver chronic-flag parser fix. If an old model was trained before the fix, its metrics and feature conclusions should be regenerated.

## Statistical Variable Selection

The engineered features are useful predictive candidates, but they are not automatically a textbook-selected statistical specification. The project now separates the modeling work into two roles:

- Statistical baseline: `databricks/14_logreg_variable_selection.py` fits a logistic-regression specification selected only on the training split using backward AIC over an interpretable full candidate model. It enforces hierarchy rules, writes selected terms, numeric collinearity diagnostics, nested likelihood-ratio tests, and coefficient/odds-ratio output.
- Predictive extensions: random forest, gradient boosting, and XGBoost use the same gold feature contract for out-of-sample discrimination, ranking, top-k lift, and calibrated probability comparison.

The selected logistic workflow centers chronic-condition count using the training-split mean before considering the quadratic term, then enforces polynomial hierarchy so the centered linear chronic term is retained whenever the centered squared term is retained. This avoids treating tree-model feature strength as a substitute for formal variable selection. The selected logistic model is the one to defend with coefficient, odds-ratio, likelihood-ratio, collinearity, and parsimony language; the tree ensembles are prediction machines.

## Modeling Implications

- Use the prospective year `t` to `t + 1` frame for final model evaluation. Same-year high-cost segmentation would leak the annual-cost target through direct cost and utilization features.
- Keep class-imbalance metrics in the primary evaluation set. The target is intentionally near 10% positive by target year, so accuracy can be misleading.
- Cost fields are right-skewed. Log transforms, tree models, and ranking metrics are appropriate.
- The local gold export has no duplicate beneficiary-year keys and no missing values in the exported fields.
- Same-year cost fields are acceptable as prior-year predictors in the prospective frame, but they should not be used to predict same-year high-cost status.
- The model is effectively learning next-year high-cost risk from prior-year utilization intensity, provider breadth, chronic burden, and cost density.
- Formal statistical claims should reference the selected logistic-regression audit, not the univariate EDA rankings or tree-model importances alone.

## Generated Artifacts

- `gold_feature_eda/row_counts_by_year.csv`
- `gold_feature_eda/missingness.csv`
- `gold_feature_eda/cost_distribution.csv`
- `gold_feature_eda/class_balance_by_target_year.csv`
- `gold_feature_eda/median_cost_by_chronic_count.csv`
- `gold_feature_eda/numeric_feature_quality.csv`
- `gold_feature_eda/feature_label_correlations.csv`
- `gold_feature_eda/annual_claim_cost_log_distribution.png`
- `gold_feature_eda/target_balance_by_year.png`
- `gold_feature_eda/top_feature_label_correlations.png`
- `gold_feature_eda/median_cost_by_chronic_count.png`
