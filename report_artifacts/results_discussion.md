# Results And Discussion

## Results

Four supervised models were evaluated for beneficiary-level next-year high-cost prediction: logistic regression, random forest, fixed-spec gradient boosting, and XGBoost. All models were trained on the locked v2 beneficiary-hash holdout split (`xxhash64_bene_id_mod_100_v2_beneficiary_hash_holdout`) and were evaluated on a held-out test set with approximately 10% positive-class prevalence. Performance was assessed using both global discrimination metrics and operational ranking metrics, including top-k capture and lift.

We do not choose the winner by training fit; we choose it by held-out ranking performance after training-workflow model selection, and we treat the held-out test set as the official final comparison.

Official comparison standard:
- beneficiary-level train/validation/test split
- training-sample model selection for tunable models
- validation sample for threshold tuning or tie-breaking
- one final comparison on the untouched held-out test sample

The final supervised comparison set includes logistic regression, random forest, gradient boosting, and XGBoost.

Gradient boosting produced the strongest held-out PR-AUC in the latest beneficiary-hash refresh. On the latest held-out test run, it achieved accuracy of 0.8740, precision of 0.4098, recall of 0.4697, AUC-ROC of 0.8333, and AUC-PR of 0.4653. More importantly for a risk-targeting use case, it identified 28.85% of all true high-cost beneficiaries within the top 5% highest-risk predictions and 43.01% within the top 10%. These results correspond to lifts of 5.77 and 4.30 over random selection, respectively.

Random forest was competitive on top-k capture but weaker on discrimination. It achieved accuracy of 0.8525, precision of 0.3207, recall of 0.4747, AUC-ROC of 0.8030, and AUC-PR of 0.3634, with top-5% and top-10% capture rates of 24.93% and 38.19%. Those top-k results are close to gradient boosting, but the lower AUC-PR and precision make it less attractive as the final balanced model.

XGBoost remained useful as a modern boosted-tree challenger, but not the best balanced model in the latest refresh. On the test set, it achieved accuracy of 0.9077, precision of 0.6843, recall of 0.2163, AUC-ROC of 0.8295, and AUC-PR of 0.4476. Its top-5% and top-10% capture rates were 29.07% and 42.31%, with lifts of 5.81 and 4.23. This profile makes XGBoost a credible sensitivity model, but it is weaker than gradient boosting on PR-AUC and slightly weaker on top-10 capture.

Logistic regression remained valuable as an interpretable baseline and was highly competitive in the beneficiary-hash refresh. It achieved accuracy of 0.8858, precision of 0.4510, recall of 0.4304, AUC-ROC of 0.8311, and AUC-PR of 0.4600. Its top-5% and top-10% capture rates were 28.96% and 43.13%. Gradient boosting improved PR-AUC slightly, while logistic regression had nearly identical top-10 capture. As a result, logistic regression is best interpreted as a strong transparent benchmark rather than a dominated model.

The full top-k capture curve reinforces this ranking. Across the operational range from 1% to 20% of the population, gradient boosting consistently delivered the strongest balanced capture and lift, random forest was almost always second, and logistic regression and XGBoost remained secondary alternatives. For example, at the top 1% cutoff, XGBoost captured 7.57% of all true positives with 7.55 lift, confirming useful concentration at the very top of the list. At the top 20% cutoff, XGBoost captured 55.43% of true positives with 2.77 lift. These numbers are useful, but they do not exceed the gradient boosting run on the main 5% and 10% operating cutoffs.

## Discussion

The results indicate that fixed-spec gradient boosting offers the best balance of discrimination, ranking quality, and generalization. Its advantage is not raw accuracy; in this imbalanced setting, accuracy is inflated by the majority class and is less informative than AUC-PR, recall, and top-k capture. Gradient boosting is therefore preferable when the objective is balanced operational identification of future high-cost claimants, while logistic regression remains defensible when the priority is maximum transparency with acceptable performance and XGBoost is useful when the priority is maximum sensitivity.

From an operational perspective, the ranking metrics are more important than threshold-dependent classification metrics. In a care-management or cost-containment setting, the key question is not whether the model can assign a hard label perfectly at one threshold, but whether it can push a disproportionate share of future high-cost beneficiaries into the highest-risk segment. By that standard, gradient boosting is clearly useful. Selecting only the top 10% of beneficiaries ranked by the model captures about 43% of all true high-cost cases, which is a substantial improvement over random targeting.

The comparison also highlights the tradeoff between interpretability, balanced predictive performance, and aggressive case-finding. Logistic regression is easier to explain and performs well on accuracy and precision, but it trails gradient boosting on AUC-PR, recall, and top-k capture. Random forest is close on top-k capture but does not improve the main discrimination metrics. XGBoost yielded the highest recall, but its precision and accuracy fell sharply and its train-to-test gaps were visible: ROC-AUC dropped from 0.8470 on train to 0.7989 on test, PR-AUC dropped from 0.4284 to 0.3617, and top-10% capture dropped from 41.60% to 36.91%. Gradient boosting retained the benefits of nonlinear modeling while delivering the strongest held-out AUC-PR and top-k targeting. That makes it the strongest balanced final model for this project.

Taken together, the evidence supports a simple modeling hierarchy. Logistic regression should be retained as the interpretable ISLP-core benchmark. Random forest should be treated as a secondary ISLP-core comparison. Fixed gradient boosting should be presented as the primary balanced model because it achieved the best rare-event ranking performance and the most convincing concentration of risk in the highest-priority beneficiary groups. XGBoost should be presented as the modern-extension, high-recall triage alternative for intervention settings where missing positives is more costly than reviewing false positives. If a supplementary holdout split influenced any selected decision threshold, it should be described as a tuning or validation split rather than as a purely passive report-only sample.

## Decision-Support Extension

The supervised results support a stronger end-to-end project story than a classifier-only presentation. In the deployed prototype, the gradient-boosting model serves as the empirical risk engine that estimates next-year high-cost probability and places each beneficiary into a risk tier. That risk output is then embedded into a discrete Markov Decision Process whose full state also includes chronic burden, utilization intensity, and prior intervention status. A tabular Q-learning algorithm is used to estimate which intervention has the highest long-run value in each state.

This extension changes the operational question from "Who is high risk?" to "Given limited intervention capacity, what should we do with that information?" That framing is closer to a real actuarial or care-management workflow. The application now demonstrates risk scoring, state classification, intervention recommendation, and side-by-side action comparison within one interface.

The limitation of this second layer must remain explicit. The supervised model is data-driven and is evaluated on observed holdout data. The RL environment is not learned from real intervention-response histories. Instead, it is a simulated policy environment built on stylized transition multipliers and reward assumptions intended to demonstrate operational decision support. For that reason, the recommended action should be interpreted as the highest-value choice within the simulated environment, not as causal evidence that the intervention will produce the outcome in real practice.

There is also an important implementation distinction inside the prototype. Offline Q-learning training bootstraps episodes from stylized tier-level risk probabilities so that the tabular policy can be learned without observed intervention trajectories. Online recommendation does not replace the beneficiary's empirical score with those stylized values. Instead, it starts from the actual supervised-model probability and maps that probability directly into the MDP state and transition logic.

The state dynamics are intentionally simplified for v1. Chronic burden is treated as fixed within an episode, while risk tier, utilization intensity, and prior intervention status evolve over time. That simplification keeps the environment readable and computationally manageable, but it also reinforces that the policy layer is a prototype simulation rather than a clinical disease-progression model.

That distinction is important to the integrity of the project. The strength of the prototype is not that it proves a treatment policy, but that it shows a coherent actuarial workflow: estimate risk, map the member into an operational state, compare interventions, and expose the recommendation through an app that makes the assumptions visible.
