# High-Cost Medicare Beneficiary Risk and Decision-Support Proposal

## Project Overview

This project studies whether structured Medicare claims data can be used to identify beneficiaries who are likely to become high-cost patients and to support downstream care-management decisions. The project is framed as an actuarial decision-support prototype: first, a supervised machine learning model estimates next-year high-cost risk; second, a simplified policy layer uses the risk score and beneficiary state information to recommend a care-management action under limited resources.

The project follows a question-to-methods structure. The central applied question is not simply whether a classifier can achieve high accuracy. The more important question is whether beneficiary-level claims, enrollment, demographic, and chronic condition features can create a useful ranked list of members for operational targeting. In a realistic care-management setting, an organization may only have capacity to contact or manage a small percentage of beneficiaries. For that reason, ranking quality, top-k capture, lift, and policy value are more relevant than accuracy alone.

## Data Description

The data source is the CMS DE-SynPUF synthetic Medicare claims dataset. This is a public synthetic dataset designed to resemble Medicare claims while protecting privacy. The project uses five major source tables:

- Beneficiary summary files, which contain beneficiary identifiers, year, demographic fields, enrollment information, and chronic condition indicators.
- Inpatient claims, which describe hospital claim activity and paid amounts.
- Outpatient claims, which describe facility-based outpatient service use and paid amounts.
- Carrier claims, which describe professional and physician-service claims.
- Prescription drug event files, which describe prescription drug use and drug cost.

The raw files are ingested into object storage and processed through a Databricks medallion pipeline. Bronze tables preserve the landed source structure, silver tables standardize entity-level fields, and the gold table aggregates the data into one row per beneficiary-year. The beneficiary-year unit is useful because it creates a stable modeling table, avoids noisy claim-line-level prediction, and supports annual risk segmentation.

The current gold table includes demographic features, enrollment features, chronic condition burden, utilization counts, cost measures, provider measures, and normalized rates. Examples include age band, sex, race code, state code, enrollment month count, chronic condition count, inpatient claim count, outpatient claim count, carrier claim count, prescription drug event count, unique provider count, total claim days, annual claim cost, cost per enrollment month, claims per enrollment month, provider fragmentation, claim-type flags, log-transformed cost measures, and cost shares by claim category.

## Target Definition

The main prediction target is high-cost beneficiary status. For each beneficiary-year, annual claim cost is computed as the sum of inpatient, outpatient, carrier, and prescription drug costs. A beneficiary-year is labeled high cost if its annual claim cost is above the 90th percentile of annual claim cost in the training data.

The 90th percentile threshold is intentionally computed only from the training split. This avoids leaking information from the validation or test data into the target definition. The expected positive class prevalence is approximately 10%, which creates an imbalanced classification problem. Because of that imbalance, the project will rely on discrimination and ranking metrics rather than accuracy alone.

## Research Question 1: Can Claims Features Predict Future High-Cost Risk?

The first research question is: can current-year beneficiary characteristics and claims history predict whether a beneficiary will be high cost in the next year?

This question motivates a supervised binary classification approach. The planned modeling path begins with logistic regression as an interpretable baseline, followed by tree-based methods such as random forest, gradient boosting, and XGBoost. Logistic regression gives a transparent benchmark and helps identify linear associations. Tree-based models are expected to capture nonlinear relationships and interactions, such as the combined effect of chronic burden, prior inpatient utilization, prescription cost, and provider fragmentation.

The main evaluation metrics will include ROC-AUC, precision-recall AUC, precision, recall, and confusion-matrix summaries. Because the target class is rare, precision-recall AUC is especially important. A model can look strong by accuracy while still performing poorly on the high-cost group, so the evaluation will focus on whether the model separates high-risk beneficiaries from the general population.

## Research Question 2: Which Beneficiary Features Are Associated With High-Cost Risk?

The second research question is: which types of beneficiary features are most associated with high-cost risk?

This question connects prediction to interpretation. Potential explanatory areas include chronic condition count, age band, prior inpatient use, prescription drug cost, total claim count, total claim days, provider count, provider fragmentation, and the share of cost coming from inpatient, outpatient, carrier, or prescription drug claims.

Several methods can be used to answer this question. Logistic regression coefficients can provide a baseline directional interpretation after suitable preprocessing. Tree-based feature importance can show which engineered features contribute most to prediction. Model comparison can also show whether nonlinear methods add value beyond an interpretable linear baseline. If time permits, permutation importance or SHAP-style explanations could be used to compare global feature importance with individual beneficiary-level explanations.

## Research Question 3: How Useful Are Predictions For Operational Targeting?

The third research question is: if the model ranks beneficiaries by predicted high-cost risk, how many true high-cost beneficiaries can be captured among the highest-risk groups?

This question is central to the project because care-management resources are usually limited. A model does not need to perfectly classify every beneficiary to be useful. It needs to concentrate future high-cost cases near the top of the ranked list so that outreach or review can be focused where it is most likely to matter.

The primary methods for this question are top-k capture and lift analysis. For example, the project can evaluate what share of all true high-cost beneficiaries appear in the top 1%, 5%, 10%, or 20% of predicted risk scores. Lift compares this capture rate against random selection. If the top 10% predicted-risk group contains far more than 10% of true high-cost beneficiaries, the model provides operational targeting value.

This analysis also helps choose between models. A model with slightly better accuracy may be less useful than one with better top-k capture. For this application, the preferred model should be the one that ranks beneficiaries most effectively in the high-risk tail.

## Research Question 4: Can Risk Scores Support Intervention Decisions?

The fourth research question is: once high-risk beneficiaries are identified, how can the risk information support a care-management action recommendation?

This question extends the project beyond prediction into decision support. The proposed policy layer uses a simplified Markov Decision Process with discrete states. A state can include risk tier, chronic burden, utilization intensity, and prior intervention status. Possible actions include no action, low-touch outreach, a care-coordination call, and intensive case management.

Because the CMS synthetic claims data does not contain real intervention-response histories, this layer should not be presented as causal treatment-effect learning. Instead, it is a simulated operational prototype. A tabular Q-learning algorithm can estimate which action has the highest long-run value under stylized transition and reward assumptions. The supervised model is empirically trained on observed data, while the reinforcement-learning layer is a policy simulation that demonstrates how risk scores could be incorporated into a decision workflow.

## Expected Methods

The expected methods include distributed data processing, supervised machine learning, model evaluation, and a simplified reinforcement-learning policy model. Databricks and Spark are used for data preparation because the project is built around a medallion architecture. The gold beneficiary-year feature table serves as the contract between data engineering and modeling.

For supervised learning, the project will compare logistic regression, random forest, gradient boosting, and XGBoost. The data will be split at the beneficiary level into training, validation, and test sets to reduce leakage across years or repeated records. Hyperparameter tuning can be performed using cross-validation within the training data, while the validation set can support threshold selection or tie-breaking. The held-out test set should be reserved for final comparison.

For evaluation, the project will report ROC-AUC, precision-recall AUC, precision, recall, top-k capture, and lift. The top-k metrics are especially aligned with the research question because they measure the model's ability to prioritize beneficiaries under resource constraints.

For the decision-support extension, the project will use a discretized MDP and tabular Q-learning. The policy state will be derived from the supervised risk score and selected beneficiary features. The policy output should be interpreted as a recommendation within a simulated environment, not as clinical or causal evidence.

## Data Pipeline and Feature Engineering Plan

The data pipeline begins with raw CMS files in object storage. These files are registered in the bronze layer, standardized in the silver layer, and aggregated in the gold layer. The gold table is the main modeling table and has one row per beneficiary-year.

Important feature groups include:

- Demographics: age band, sex, race code, and state code.
- Enrollment: enrollment month count and normalized rates per enrollment month.
- Chronic burden: count of chronic condition indicators.
- Utilization: inpatient, outpatient, carrier, and prescription drug claim counts.
- Cost: total annual cost, claim-type total costs, log-transformed costs, and claim-type cost shares.
- Care pattern features: unique provider count, provider fragmentation index, and multiple-provider flag.
- Binary utilization flags: indicators for any inpatient, outpatient, carrier, or prescription drug activity.

The feature engineering plan emphasizes defensible timing. Features should be computed from the current or prior year, while the target should represent high-cost status in the next year when possible. If the available project version uses same-year target construction for demonstration, the limitation should be clearly stated and the future-risk framing should be treated as the intended modeling design.

## Evaluation Plan

The evaluation plan has three levels. First, global discrimination metrics will measure whether the model separates high-cost from non-high-cost beneficiary-years. Second, ranking metrics will measure whether the highest-risk predictions capture a disproportionate share of true high-cost cases. Third, the decision-support layer will be evaluated by comparing simulated long-run value across available actions.

The final supervised model should not be selected by training performance. It should be selected by out-of-sample performance after tuning. A strong model should show good test-set ROC-AUC and precision-recall AUC, but it should also show high top-k capture and lift. In the current project artifacts, gradient boosting is the preferred model because it provides the strongest overall ranking performance on the held-out test set.

For the policy layer, evaluation will focus on whether the learned policy produces sensible action differences across risk tiers and utilization states. A low-risk beneficiary should not receive the same recommendation as a very high-risk, high-utilization beneficiary unless the learned value estimates justify that result. The policy layer should also expose its assumptions so that users can distinguish empirical prediction from simulated decision logic.

## Expected Contributions

This project contributes a complete end-to-end workflow rather than a standalone model. It connects raw claims ingestion, distributed feature engineering, supervised model training, model evaluation, and application-facing decision support.

The main expected contribution is an empirically evaluated risk engine for identifying likely high-cost beneficiaries. A second contribution is the operational evaluation framing: top-k capture and lift directly answer whether the model is useful for prioritizing limited care-management resources. A third contribution is the policy prototype, which demonstrates how a risk score can feed into a structured intervention recommendation workflow.

## Limitations and Risks

Several limitations must be stated clearly. First, the CMS DE-SynPUF dataset is synthetic, so results should not be interpreted as real-world Medicare performance. Second, claims data is observational, so the supervised model estimates risk rather than causal effects. Third, high-cost status is imbalanced, which means accuracy can be misleading. Fourth, feature timing must be carefully controlled to avoid leakage.

The largest limitation is the policy layer. The MDP and Q-learning extension is built on stylized transition and reward assumptions because the dataset does not contain randomized interventions or real intervention-response histories. Therefore, the recommended action is not proof that the action would reduce cost or improve outcomes in reality. It is a simulated decision-support result under explicit assumptions.

## Conclusion

The project asks whether CMS synthetic Medicare claims can support high-cost beneficiary prediction and operational decision support. The planned methods follow directly from the research questions: distributed feature engineering creates a beneficiary-year modeling table; supervised classification estimates high-cost risk; top-k capture and lift evaluate targeting usefulness; and a simulated MDP/Q-learning layer explores how predicted risk could support intervention recommendations.

This structure keeps the project aligned with course methods while also making the application realistic. The final result should show not only whether a model can predict high-cost status, but also whether the prediction is useful for prioritizing beneficiaries and informing care-management decisions under limited resources.
