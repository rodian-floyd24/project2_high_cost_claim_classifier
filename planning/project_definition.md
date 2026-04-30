# Project Definition

## Title

Actuarial Decision-Support Prototype For High-Cost Beneficiary Management

## Core Question

Given structured beneficiary and claims features, what is the beneficiary's next-year high-cost risk, and which intervention has the highest estimated long-run value under a stylized care-management policy model?

## Recommended Unit of Analysis

Use `one row = one beneficiary-year`.

This is the safest formulation for the CMS synthetic claims project because it:

- turns multiple raw claim tables into one stable modeling table
- creates a realistic annual risk-segmentation use case
- avoids very noisy claim-line level modeling
- makes the target and feature timing easier to defend

## Target Variable

For each beneficiary-year:

- `annual_claim_cost = sum(allowed_or_paid_amount across the beneficiary-year)`
- `high_cost = 1 if annual_claim_cost > Q0.90 else 0`

`Q0.90` must be computed from the training split only.

## Modeling Path

1. Logistic regression benchmark
2. Gradient boosting primary risk engine
3. Discretized MDP policy layer fed by the supervised risk score
4. Tabular Q-learning for intervention recommendation

## Primary Metrics

- ROC-AUC
- AUC-PR
- Top-k capture and lift
- Policy comparison by estimated long-run value within the simulated environment

## Business Framing

This project is an actuarial decision-support tool. The supervised layer identifies beneficiaries likely to enter the high-cost tail, and the policy layer uses that risk signal to recommend how limited care-management resources should be allocated over time.

## Limitation Statement

The risk engine is trained on observed beneficiary data. The MDP/Q-learning layer is a simulation prototype built on stylized transition and reward assumptions rather than causal intervention-response histories. It should be presented as operational policy design under assumptions, not as validated treatment-effect learning.
