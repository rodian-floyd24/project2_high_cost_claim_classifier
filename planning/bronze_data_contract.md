# Bronze Data Contract

This file freezes the structurally important first-pass fields for bronze validation.

## Beneficiary Summary

- entity: `beneficiary_summary`
- expected_primary_key: `DESYNPUF_ID`
- expected_date_column: none at the row grain
- expected_cost_column: `MEDREIMB_IP` as a beneficiary summary reimbursement reference field, not the target cost source

## Inpatient Claims

- entity: `inpatient_claims`
- expected_primary_key: claim identifier column if present, plus `DESYNPUF_ID` as beneficiary join key
- expected_date_column: `CLM_THRU_DT`
- expected_cost_column: `CLM_PMT_AMT`

## Outpatient Claims

- entity: `outpatient_claims`
- expected_primary_key: claim identifier column if present, plus `DESYNPUF_ID` as beneficiary join key
- expected_date_column: `CLM_THRU_DT`
- expected_cost_column: `CLM_PMT_AMT`

## Carrier Claims

- entity: `carrier_claims`
- expected_primary_key: claim or line identifier column if present, plus `DESYNPUF_ID` as beneficiary join key
- expected_date_column: `CLM_THRU_DT`
- expected_cost_column: `LINE_NCH_PMT_AMT`

## Prescription Drug Events

- entity: `prescription_drug_events`
- expected_primary_key: PDE identifier column if present, plus `DESYNPUF_ID` as beneficiary join key
- expected_date_column: `SRVC_DT`
- expected_cost_column: `TOT_RX_CST_AMT`

## First-Pass Cost Convention

Version 1 will use the most consistently populated paid amount style field available in each raw claim entity:

- inpatient: `CLM_PMT_AMT`
- outpatient: `CLM_PMT_AMT`
- carrier: `LINE_NCH_PMT_AMT`
- PDE: `TOT_RX_CST_AMT`

This convention must remain fixed through bronze, silver, gold, and the first logistic-regression baseline.
