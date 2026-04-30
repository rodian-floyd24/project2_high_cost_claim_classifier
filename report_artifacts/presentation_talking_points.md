# Presentation Talking Points

## Recommended Framing

- Risk engine: empirically trained supervised model
- Policy layer: simulated MDP/Q-learning intervention prototype
- Recommendation: action with highest estimated long-run value under stylized assumptions
- We do not choose the winner by training fit; we choose it by held-out ranking performance after training-workflow model selection, and we treat the held-out test set as the official final comparison.

## What To Say

- The supervised model predicts next-year high-cost risk at the beneficiary-year level.
- Official supervised-model standard: beneficiary-level train/validation/test split, training-sample model selection for tunable models, validation-based threshold tuning, and one final comparison on the untouched held-out test sample.
- The final supervised comparison set includes logistic regression, random forest, gradient boosting, and XGBoost.
- Fixed gradient boosting is the primary deployed risk engine because it led the held-out comparison on AUC-ROC, AUC-PR, and top-k targeting metrics.
- XGBoost is the high-recall triage alternative: it catches more positives but produces substantially more false positives, so it is not the best balanced model.
- The app does not stop at prediction. It uses the predicted risk tier as part of an operational state for intervention selection.
- The policy layer demonstrates how an actuarial or care-management team might translate risk scores into action under limited resources.

## What Not To Say

- Do not say the RL layer is causally validated.
- Do not say the recommendation proves the intervention will reduce cost in practice.
- Do not describe the policy explanation as strict interpretability of treatment effects.

## Required Limitation Statement

The supervised risk engine is trained on observed data. The MDP/Q-learning layer is a simulated decision environment built on stylized transition and reward assumptions rather than real intervention-response histories. Recommendations therefore reflect estimated long-run value inside the prototype environment, not validated causal treatment effects.
