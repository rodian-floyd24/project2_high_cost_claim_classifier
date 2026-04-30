# Final Results Table

Table 1 summarizes the final out-of-sample test-set results for the candidate models on the next-year high-cost beneficiary prediction task.

| Model | Group | Accuracy | Error | Precision | Recall | AUC-ROC | AUC-PR | Top 5% Capture | Top 5% Lift | Top 10% Capture | Top 10% Lift |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gradient boosting | ISLP core | 0.8533 | 0.1467 | 0.3329 | 0.4769 | 0.8099 | 0.3879 | 0.2511 | 5.0207 | 0.3877 | 3.8767 |
| Logistic regression | ISLP core | 0.8603 | 0.1397 | 0.3432 | 0.4469 | 0.8060 | 0.3634 | 0.2386 | 4.7708 | 0.3799 | 3.7994 |
| Random forest | ISLP core | 0.8525 | 0.1475 | 0.3207 | 0.4747 | 0.8030 | 0.3634 | 0.2493 | 4.9859 | 0.3819 | 3.8192 |
| XGBoost | modern extension | 0.7131 | 0.2869 | 0.2179 | 0.7145 | 0.7989 | 0.3617 | 0.2322 | 4.6406 | 0.3691 | 3.6897 |

Notes:
- All metrics are from the holdout test set with beneficiary-level train/validation/test splitting.
- The positive-class prevalence is approximately 9.6%-10.1% across the model-specific test exports.
- The high-cost label threshold was defined from the training split only; the canonical comparison export uses thresholds of `10460` for logistic regression, gradient boosting, and XGBoost, and `10540` for random forest.
- Tuning-split decision thresholds were `0.20` for logistic regression, `0.64` for random forest, and `0.50` for XGBoost. The fixed gradient boosting export did not report a tuning threshold in the canonical table. XGBoost is reported as a high-recall triage alternative rather than the primary balanced model.
