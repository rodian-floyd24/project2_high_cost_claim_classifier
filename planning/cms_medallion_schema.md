# CMS Synthetic Claims Medallion Plan

## Dataset Choice

Use CMS synthetic Medicare claims data, ideally the DE-SynPUF-style linked beneficiary and claims files.

## Why This Dataset

- claims-shaped schema rather than generic tabular data
- supports a real bronze/silver/gold pipeline story
- professional fit for insurance / actuarial analytics framing
- enough structural complexity to justify Databricks transformations

## Bronze Layer

Bronze stores raw files exactly as landed from the source.

### Bronze Objects

- `bronze/beneficiary_summary/`
- `bronze/inpatient_claims/`
- `bronze/outpatient_claims/`
- `bronze/carrier_claims/`
- `bronze/prescription_events/`
- `bronze/reference/`

### Bronze Rules

- no column renaming
- no dropped fields
- raw file checksum or ingestion timestamp captured
- preserve source file names and load date

### Suggested Bronze Metadata Columns

- `source_file_name`
- `ingested_at_utc`
- `source_dataset_version`

## Silver Layer

Silver standardizes types, cleans columns, and creates one cleaned table per source entity.

### `silver_beneficiaries`

One row per beneficiary per year.

Suggested columns:

- `bene_id`
- `year`
- `sex`
- `race_code`
- `birth_date` or derived age band
- `state_code`
- `county_code` if available
- `esrd_flag`
- `chronic_condition_flags`
- `enrollment_months_count`
- `dual_eligibility_flag` if available
- `death_flag`

### `silver_inpatient_claims`

One row per inpatient claim.

Suggested columns:

- `claim_id`
- `bene_id`
- `year`
- `admit_date`
- `discharge_date`
- `drg_code`
- `provider_id`
- `state_code`
- `total_claim_amount`
- `payment_amount`
- `claim_days`

### `silver_outpatient_claims`

One row per outpatient claim.

Suggested columns:

- `claim_id`
- `bene_id`
- `year`
- `claim_from_date`
- `claim_thru_date`
- `provider_id`
- `state_code`
- `payment_amount`
- `claim_days`

### `silver_carrier_claims`

One row per carrier / physician claim.

Suggested columns:

- `claim_id`
- `bene_id`
- `year`
- `line_count`
- `provider_specialty` if available
- `payment_amount`
- `allowed_amount` if available

### `silver_pde`

One row per prescription event.

Suggested columns:

- `pde_id`
- `bene_id`
- `year`
- `drug_fill_date`
- `drug_cost`
- `days_supply`

### Silver Rules

- standardize column names to snake_case
- coerce dates and numeric amounts
- remove exact duplicates
- keep null-handling explicit
- document any dropped columns

## Version Decision

Version 1 will be descriptive risk segmentation using the full beneficiary-year table.

That means the first working system will:

- build a stable `beneficiary-year` gold table
- train the logistic regression baseline on that stable table
- defer stricter prospective timing windows to version 2

Version 2 can tighten leakage control by restricting features to early-period or prior-period information only.

## Silver Aggregation Rules

Use a consistent paid-cost style measure across all files for version 1.

If multiple monetary fields exist, choose the most consistently populated paid amount equivalent in each file and document the exact source column in code.

First-pass beneficiary-year formulas:

- `inpatient_total_cost = sum(inpatient_paid_amount)` over `bene_id, year`
- `outpatient_total_cost = sum(outpatient_paid_amount)` over `bene_id, year`
- `carrier_total_cost = sum(carrier_paid_amount)` over `bene_id, year`
- `rx_total_cost = sum(pde_paid_amount_or_drug_cost)` over `bene_id, year`
- `annual_claim_cost = inpatient_total_cost + outpatient_total_cost + carrier_total_cost + rx_total_cost`

Utilization formulas:

- `inpatient_claim_count = count(distinct inpatient_claim_id)` over `bene_id, year`
- `outpatient_claim_count = count(distinct outpatient_claim_id)` over `bene_id, year`
- `carrier_claim_count = count(distinct carrier_claim_id)` over `bene_id, year`
- `pde_claim_count = count(distinct pde_id)` over `bene_id, year`
- `total_claim_days = sum(claim_days across inpatient and outpatient claims)` over `bene_id, year`
- `unique_provider_count = count(distinct provider_id across inpatient, outpatient, and carrier claims)` over `bene_id, year`

## Gold Layer

Gold should be the model contract. Keep it stable.

## Recommended Gold Table

### `gold_beneficiary_year_features`

One row per `bene_id, year`.

### Core Keys

- `bene_id`
- `year`

### Target Columns

- `annual_claim_cost`
- `high_cost`

### Allowed in First-Pass Model

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

### Defer Until Later

- `esrd_flag`
- `prior_year_total_cost`
- `avg_claim_cost`
- `max_single_claim_cost`
- `distinct_diagnosis_group_count`
- `distinct_procedure_group_count`
- `drug_fill_count`
- `avg_days_supply`
- `reporting_months_active`
- `acute_inpatient_flag`
- `multi_setting_utilization_flag`

## Features to Avoid

Do not use columns that would only be known after the full outcome period if you plan to present the system as prospective scoring.

Examples to avoid for a first version:

- post-outcome leakage variables
- future claim totals from the same scoring window
- directly derived target fragments

## Recommended Scoring Story

Version 1 is the stable beneficiary-year segmentation build.

Version 2 is the tighter prospective scoring build:

- use early-period or prior-period features
- predict whether the beneficiary ends the year in the top decile of annual cost
