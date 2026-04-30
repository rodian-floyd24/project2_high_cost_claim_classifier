# CMS Source Inventory

This file freezes the first-pass CMS synthetic claims ingestion plan.

## Source URL

Primary CMS landing page:

- `https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf`

Sample download page pattern:

- `https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/DESample01.html`

## Official CMS File Structure

CMS states that each DE-SynPUF sample contains eight files:

- three beneficiary summary files, one for each year
- one inpatient claims file containing 2008-2010 data
- one outpatient claims file containing 2008-2010 data
- one PDE file containing 2008-2010 data
- two carrier claims files containing 2008-2010 data, split into `A` and `B`

CMS also states that all claims for a beneficiary live in the same sample number and that `DESYNPUF_ID` is the beneficiary linking key across the files.

## First-Pass Sample Scope

Use `DE1.0 Sample 1` first to get the pipeline running before scaling out to more samples.

## First-Pass Source File Names

Beneficiary:

- `DE1_0_2008_Beneficiary_Summary_File_Sample_1`
- `DE1_0_2009_Beneficiary_Summary_File_Sample_1`
- `DE1_0_2010_Beneficiary_Summary_File_Sample_1`

Inpatient:

- `DE1_0_2008_to_2010_Inpatient_Claims_Sample_1`

Outpatient:

- `DE1_0_2008_to_2010_Outpatient_Claims_Sample_1`

Carrier:

- `DE1_0_2008_to_2010_Carrier_Claims_Sample_1A`
- `DE1_0_2008_to_2010_Carrier_Claims_Sample_1B`

Prescription Drug Events:

- `DE1_0_2008_to_2010_Prescription_Drug_Events_Sample_1`

## File-to-Entity Mapping

- beneficiary -> 2008, 2009, 2010 beneficiary summary files
- inpatient -> 2008-2010 inpatient claims file
- outpatient -> 2008-2010 outpatient claims file
- carrier -> 2008-2010 carrier claims files `1A` and `1B`
- PDE -> 2008-2010 prescription drug events file

## Expected Record Counts

Full DE-SynPUF population, per CMS:

- beneficiary summary: 2,326,856 in 2008; 2,291,320 in 2009; 2,255,098 in 2010
- inpatient claims: 547,800 in 2008; 504,941 in 2009; 280,081 in 2010
- outpatient claims: 5,673,808 in 2008; 6,519,340 in 2009; 3,633,839 in 2010
- carrier claims: 34,276,324 in 2008; 37,304,993 in 2009; 23,282,135 in 2010
- PDE: 39,927,827 in 2008; 43,379,293 in 2009; 27,778,849 in 2010

Sample 1 counts from the CMS user manual:

- beneficiary 2008: 116,352
- beneficiary 2009: 114,538
- beneficiary 2010: 112,754
- inpatient 2008-2010 claims: 66,773
- outpatient 2008-2010 claims: 790,790
- carrier 2008-2010 claims total: 4,741,335
- carrier sample 1A: 2,370,667
- carrier sample 1B: 2,370,668
- PDE 2008-2010 events: 5,552,421

## Join Keys

Primary beneficiary join key:

- `DESYNPUF_ID`

Recommended working surrogate names after silver cleanup:

- `bene_id` <- `DESYNPUF_ID`
- `claim_id` from the relevant claim identifier where present

## First-Pass Ingestion Decision

Version 1 will ingest only `DE1.0 Sample 1` to get the end-to-end pipeline running quickly.

Version 2 can scale to additional samples after the bronze/silver/gold pipeline is stable.
