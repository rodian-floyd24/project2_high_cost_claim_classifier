# Demo Script

Use this exact sentence consistently during the demo:

The risk engine is empirically trained on observed data, while the reinforcement-learning policy layer is a simulated decision prototype built on stylized transition and reward assumptions.

## Scenario 1: Low-risk routine beneficiary

Profile summary:
- Female, age 65-74, Florida
- 1 chronic condition
- Light outpatient and carrier utilization
- No recent intervention

Predicted risk probability:
- `0.0068`

Risk tier:
- `Lower`

MDP state:
- `Low risk, low chronic burden, medium utilization, no recent intervention`
- State ID: `3`

Recommended action:
- `Low-touch outreach`

Short rationale:
- The beneficiary is low risk with low chronic burden and medium utilization. Low-touch outreach offers a favorable low-cost intervention under the current policy.

What the audience should notice:
- The empirical risk score is very low.
- The MDP state still captures operational context beyond risk alone.
- The empirical risk score is very low, but the beneficiary still exhibits enough utilization activity that the simulated policy assigns a low-cost outreach action rather than full inaction.
- This illustrates that the policy layer is optimizing operational value under stylized assumptions, not simply mirroring the classifier threshold.

## Scenario 2: Moderate chronic-care beneficiary

Profile summary:
- Male, age 75-84, Texas
- 4 chronic conditions
- Moderate claims volume with meaningful utilization and cost burden
- Recent low-touch outreach already occurred

Predicted risk probability:
- `0.0641`

Risk tier:
- `Lower`

MDP state:
- `Low risk, medium chronic burden, high utilization, recent low-touch outreach`
- State ID: `16`

Recommended action:
- `Care coordination call`

Short rationale:
- The beneficiary is low risk with medium chronic burden and high utilization. Care coordination provides the best balance of intervention cost and expected downstream risk reduction.

What the audience should notice:
- This case is useful because the empirical risk score is still below the high-risk cutoffs.
- The MDP state surfaces chronic burden, utilization intensity, and prior intervention history.
- The policy layer treats this as an operational management case rather than a pure high-risk prediction case, which is why the recommended action is stronger than low-touch outreach but still below intensive management.

## Scenario 3: Very-high-risk complex beneficiary

Profile summary:
- Female, age 85+, New York
- 7 chronic conditions
- High inpatient, outpatient, carrier, and prescription utilization
- No recent intervention

Predicted risk probability:
- `0.6408`

Risk tier:
- `Very high`

MDP state:
- `Very high risk, high chronic burden, high utilization, no recent intervention`
- State ID: `105`

Recommended action:
- `Care coordination call`

Short rationale:
- The beneficiary is very_high risk with high chronic burden and high utilization. Care coordination provides the best balance of intervention cost and expected downstream risk reduction.

What the audience should notice:
- This is the cleanest example of the empirical risk engine identifying a truly elevated-risk member.
- The MDP state and policy recommendation show how the project moves from prediction to action, with the strongest intervention reserved for the highest-acuity case.
- The policy recommendation should be described as simulated operational decision support, not causal treatment guidance.

## Demo Closing

Use this wrap-up:

- Risk engine: empirically trained supervised model
- MDP state: discrete operational state used by the policy layer
- Policy recommendation: highest-value action under the simulated environment
- Simulated RL disclaimer: recommendations reflect stylized transition and reward assumptions, not validated intervention effects
