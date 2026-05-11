# Final Results Table

Table 1 summarizes the final out-of-sample test-set results for the candidate models on the next-year high-cost beneficiary prediction task.

| Model | Group | Accuracy | Error | Precision | Recall | AUC-ROC | AUC-PR | Top 5% Capture | Top 5% Lift | Top 10% Capture | Top 10% Lift |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gradient boosting | ISLP core | 0.8740 | 0.1260 | 0.4098 | 0.4697 | 0.8333 | 0.4653 | 0.2885 | 5.7687 | 0.4301 | 4.3011 |
| Logistic regression | ISLP core | 0.8858 | 0.1142 | 0.4510 | 0.4304 | 0.8311 | 0.4600 | 0.2896 | 5.7913 | 0.4313 | 4.3124 |
| Random forest | ISLP core | 0.8837 | 0.1163 | 0.4420 | 0.4341 | 0.8324 | 0.4585 | 0.2876 | 5.7518 | 0.4276 | 4.2756 |
| XGBoost | modern extension | 0.9077 | 0.0923 | 0.6843 | 0.2163 | 0.8295 | 0.4476 | 0.2907 | 5.8140 | 0.4231 | 4.2304 |

Notes:
- All metrics are from the holdout test set with the locked v2 beneficiary-hash split: `xxhash64_bene_id_mod_100_v2_beneficiary_hash_holdout`.
- The positive-class prevalence is approximately 10.44% in the beneficiary-hash test split.
- The high-cost label threshold was defined from the training split only; the canonical comparison export uses threshold `10420`.
- Tuning-split decision thresholds were `0.25` for logistic regression, `0.27` for random forest, and `0.50` for XGBoost. Gradient boosting is selected for strongest PR-AUC with competitive top-k capture; logistic regression remains a highly competitive interpretable baseline.
