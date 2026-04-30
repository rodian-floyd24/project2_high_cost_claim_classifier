# Immediate Next Steps

## 1. Validate Dataset Access

Confirm the exact CMS synthetic files you will use and where they will be downloaded from.

Output:

- source URL list
- file names
- expected row counts
- expected join keys

## 2. Freeze the Row Definition

Use:

- `one row = one beneficiary-year`

Output:

- short note explaining why beneficiary-year is the modeling unit

## 3. Freeze the Gold Schema

Start with these minimum columns:

- `bene_id`
- `year`
- `annual_claim_cost`
- `high_cost`
- `age_band`
- `sex`
- `race_code`
- `state_code`
- `enrollment_months_count`
- `chronic_condition_count`
- `inpatient_claim_count`
- `outpatient_claim_count`
- `carrier_claim_count`
- `pde_claim_count`
- `total_claim_days`
- `unique_provider_count`
- `rx_total_cost`
- `inpatient_total_cost`
- `outpatient_total_cost`
- `carrier_total_cost`

Do not expand beyond this list for version 1.

## 4. Build the Repo Before Logic Sprawls

Create and keep these directories:

- `data_ingestion/`
- `databricks/`
- `backend/`
- `frontend/`
- `src/`
- `planning/`

## 5. First Build Target

The first working milestone is not the UI.

The first working milestone is:

- raw CMS files landed in object storage
- silver cleaned claims tables produced
- one gold beneficiary-year table materialized

## 6. First Modeling Target

Train only the logistic regression baseline first.

Modeling sequence:

- create train / validation / test split
- compute `Q0.90` on training only
- create `high_cost` using that threshold
- fit logistic regression
- log metrics and artifacts to MLflow

Do not tune boosting models until:

- train / validation split is stable
- target definition is final
- gold schema stops changing
