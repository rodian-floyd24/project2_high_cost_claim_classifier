# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer for CMS Synthetic Claims
# MAGIC
# MAGIC Reads the bronze Delta tables, standardizes types and keys, and writes one clean silver table per entity plus a run audit table.

# COMMAND ----------

from __future__ import annotations

import os
from datetime import datetime, timezone
from uuid import uuid4

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window


BRONZE_DATABASE = os.environ.get("BRONZE_DATABASE", "default")
SILVER_DATABASE = os.environ.get("SILVER_DATABASE", BRONZE_DATABASE)
AUDIT_TABLE_NAME = "silver_audit_summary"

BRONZE_TABLES = {
    "beneficiary_summary": "bronze_beneficiary_summary",
    "inpatient_claims": "bronze_inpatient_claims",
    "outpatient_claims": "bronze_outpatient_claims",
    "carrier_claims": "bronze_carrier_claims",
    "prescription_drug_events": "bronze_pde",
}

SILVER_TABLES = {
    "beneficiary_summary": "silver_beneficiaries",
    "inpatient_claims": "silver_inpatient_claims",
    "outpatient_claims": "silver_outpatient_claims",
    "carrier_claims": "silver_carrier_claims",
    "prescription_drug_events": "silver_pde",
}

DEDUPE_KEYS = {
    "beneficiary_summary": ["bene_id", "year"],
    "inpatient_claims": ["claim_id", "bene_id", "year", "claim_segment"],
    "outpatient_claims": ["claim_id", "bene_id", "year", "claim_segment"],
    "carrier_claims": ["claim_id", "bene_id", "year"],
    "prescription_drug_events": ["pde_id", "bene_id", "year"],
}

CHRONIC_FLAG_COLUMNS = {
    "alzheimers_flag": "SP_ALZHDMTA",
    "chf_flag": "SP_CHF",
    "chronic_kidney_disease_flag": "SP_CHRNKIDN",
    "cancer_flag": "SP_CNCR",
    "copd_flag": "SP_COPD",
    "depression_flag": "SP_DEPRESSN",
    "diabetes_flag": "SP_DIABETES",
    "ischemic_heart_disease_flag": "SP_ISCHMCHT",
    "osteoporosis_flag": "SP_OSTEOPRS",
    "rheumatoid_arthritis_oa_flag": "SP_RA_OA",
    "stroke_tia_flag": "SP_STRKETIA",
}


def read_bronze(table_name: str) -> DataFrame:
    full_name = f"{BRONZE_DATABASE}.{table_name}"
    if not spark.catalog.tableExists(full_name):
        raise ValueError(f"Required bronze table does not exist: {full_name}")
    return spark.table(full_name)


def nullify_blank(column_name: str) -> Column:
    value = F.trim(F.col(column_name).cast("string"))
    return F.when((value == "") | value.isNull(), None).otherwise(value)


def parse_date(column_name: str) -> Column:
    return F.to_date(nullify_blank(column_name), "yyyyMMdd")


def parse_int(column_name: str) -> Column:
    return nullify_blank(column_name).cast("int")


def parse_double(column_name: str) -> Column:
    return nullify_blank(column_name).cast("double")


def yes_no_flag(column_name: str) -> Column:
    value = F.upper(nullify_blank(column_name))
    return (
        F.when(value.isin("Y", "YES", "TRUE", "T", "1"), F.lit(True))
        .when(value.isin("N", "NO", "FALSE", "F", "0", "2"), F.lit(False))
        .otherwise(F.lit(None).cast("boolean"))
    )


def date_parse_failed(column_name: str) -> Column:
    raw = nullify_blank(column_name)
    return raw.isNotNull() & F.to_date(raw, "yyyyMMdd").isNull()


def int_parse_failed(column_name: str) -> Column:
    raw = nullify_blank(column_name)
    return raw.isNotNull() & raw.cast("int").isNull()


def double_parse_failed(column_name: str) -> Column:
    raw = nullify_blank(column_name)
    return raw.isNotNull() & raw.cast("double").isNull()


def first_non_null(columns: list[str]) -> Column:
    return F.coalesce(*[nullify_blank(column_name) for column_name in columns])


def sum_columns(columns: list[str]) -> Column:
    # Missing line amounts do not contribute to claim totals; malformed values are audited separately.
    total = F.lit(0.0)
    for column_name in columns:
        total = total + F.coalesce(parse_double(column_name), F.lit(0.0))
    return total


def count_non_null(columns: list[str]) -> Column:
    total = F.lit(0)
    for column_name in columns:
        total = total + F.when(nullify_blank(column_name).isNotNull(), F.lit(1)).otherwise(F.lit(0))
    return total


def count_true(columns: list[Column]) -> Column:
    total = F.lit(0)
    for column in columns:
        total = total + F.when(column, F.lit(1)).otherwise(F.lit(0))
    return total


def add_processing_metadata(df: DataFrame, source_table: str, run_ts: Column) -> DataFrame:
    return (
        df.withColumn("_silver_source_table", F.lit(source_table))
        .withColumn("_silver_processed_at_utc", run_ts)
    )


def row_hash(df: DataFrame) -> Column:
    excluded_prefixes = ("_bronze_", "_silver_")
    hashed_columns = [
        F.coalesce(F.col(column_name).cast("string"), F.lit("__NULL__"))
        for column_name in sorted(df.columns)
        if not column_name.startswith(excluded_prefixes)
    ]
    return F.sha2(F.concat_ws("||", *hashed_columns), 256)


def deduplicate_latest(df: DataFrame, key_columns: list[str]) -> DataFrame:
    with_hash = df.withColumn("_dedupe_row_hash", row_hash(df))
    window = Window.partitionBy(*[F.col(column_name) for column_name in key_columns]).orderBy(
        F.col("_bronze_loaded_at_utc").desc_nulls_last(),
        F.col("_bronze_source_file").desc_nulls_last(),
        F.col("_dedupe_row_hash").desc(),
    )
    return with_hash.withColumn("_dedupe_row_number", F.row_number().over(window)).filter(
        F.col("_dedupe_row_number") == 1
    ).drop("_dedupe_row_hash", "_dedupe_row_number")


def build_beneficiaries(df: DataFrame, run_ts: Column) -> DataFrame:
    chronic_flag_exprs = [yes_no_flag(source_name) for source_name in CHRONIC_FLAG_COLUMNS.values()]
    parse_failure_exprs = [
        date_parse_failed("BENE_BIRTH_DT"),
        date_parse_failed("BENE_DEATH_DT"),
        int_parse_failed("BENE_SEX_IDENT_CD"),
        int_parse_failed("BENE_HI_CVRAGE_TOT_MONS"),
        int_parse_failed("BENE_SMI_CVRAGE_TOT_MONS"),
        int_parse_failed("BENE_HMO_CVRAGE_TOT_MONS"),
        int_parse_failed("PLAN_CVRG_MOS_NUM"),
        double_parse_failed("MEDREIMB_IP"),
        double_parse_failed("MEDREIMB_OP"),
        double_parse_failed("MEDREIMB_CAR"),
        double_parse_failed("BENRES_IP"),
        double_parse_failed("BENRES_OP"),
        double_parse_failed("BENRES_CAR"),
        double_parse_failed("PPPYMT_IP"),
        double_parse_failed("PPPYMT_OP"),
        double_parse_failed("PPPYMT_CAR"),
    ]

    cleaned = df.select(
        nullify_blank("DESYNPUF_ID").alias("bene_id"),
        F.col("_bronze_source_year").cast("int").alias("year"),
        parse_date("BENE_BIRTH_DT").alias("birth_date"),
        parse_date("BENE_DEATH_DT").alias("death_date"),
        parse_int("BENE_SEX_IDENT_CD").alias("sex_code"),
        F.when(parse_int("BENE_SEX_IDENT_CD") == 1, F.lit("male"))
        .when(parse_int("BENE_SEX_IDENT_CD") == 2, F.lit("female"))
        .otherwise(F.lit("unknown"))
        .alias("sex"),
        nullify_blank("BENE_RACE_CD").alias("race_code"),
        yes_no_flag("BENE_ESRD_IND").alias("esrd_flag"),
        nullify_blank("SP_STATE_CODE").alias("state_code"),
        nullify_blank("BENE_COUNTY_CD").alias("county_code"),
        parse_int("BENE_HI_CVRAGE_TOT_MONS").alias("hi_coverage_months"),
        parse_int("BENE_SMI_CVRAGE_TOT_MONS").alias("smi_coverage_months"),
        parse_int("BENE_HMO_CVRAGE_TOT_MONS").alias("hmo_coverage_months"),
        parse_int("PLAN_CVRG_MOS_NUM").alias("plan_coverage_months"),
        *[
            yes_no_flag(source_name).alias(target_name)
            for target_name, source_name in CHRONIC_FLAG_COLUMNS.items()
        ],
        count_true(chronic_flag_exprs).alias("chronic_condition_count"),
        parse_double("MEDREIMB_IP").alias("beneficiary_inpatient_reimbursement_amt"),
        parse_double("MEDREIMB_OP").alias("beneficiary_outpatient_reimbursement_amt"),
        parse_double("MEDREIMB_CAR").alias("beneficiary_carrier_reimbursement_amt"),
        parse_double("BENRES_IP").alias("beneficiary_inpatient_responsibility_amt"),
        parse_double("BENRES_OP").alias("beneficiary_outpatient_responsibility_amt"),
        parse_double("BENRES_CAR").alias("beneficiary_carrier_responsibility_amt"),
        parse_double("PPPYMT_IP").alias("beneficiary_inpatient_primary_payer_amt"),
        parse_double("PPPYMT_OP").alias("beneficiary_outpatient_primary_payer_amt"),
        parse_double("PPPYMT_CAR").alias("beneficiary_carrier_primary_payer_amt"),
        count_true(parse_failure_exprs).alias("_parse_failure_count"),
        F.col("_bronze_source_file").alias("_bronze_source_file"),
        F.col("_bronze_loaded_at_utc").alias("_bronze_loaded_at_utc"),
    )

    return add_processing_metadata(
        cleaned.withColumn("death_flag", F.col("death_date").isNotNull()),
        BRONZE_TABLES["beneficiary_summary"],
        run_ts,
    )


def build_inpatient_claims(df: DataFrame, run_ts: Column) -> DataFrame:
    parse_failure_exprs = [
        int_parse_failed("SEGMENT"),
        date_parse_failed("CLM_FROM_DT"),
        date_parse_failed("CLM_THRU_DT"),
        date_parse_failed("CLM_ADMSN_DT"),
        date_parse_failed("NCH_BENE_DSCHRG_DT"),
        double_parse_failed("CLM_PMT_AMT"),
        double_parse_failed("NCH_PRMRY_PYR_CLM_PD_AMT"),
        double_parse_failed("CLM_PASS_THRU_PER_DIEM_AMT"),
        double_parse_failed("NCH_BENE_IP_DDCTBL_AMT"),
        double_parse_failed("NCH_BENE_PTA_COINSRNC_LBLTY_AM"),
        double_parse_failed("NCH_BENE_BLOOD_DDCTBL_LBLTY_AM"),
        int_parse_failed("CLM_UTLZTN_DAY_CNT"),
    ]

    cleaned = df.select(
        nullify_blank("CLM_ID").alias("claim_id"),
        nullify_blank("DESYNPUF_ID").alias("bene_id"),
        parse_int("SEGMENT").alias("claim_segment"),
        parse_date("CLM_FROM_DT").alias("claim_from_date"),
        parse_date("CLM_THRU_DT").alias("claim_thru_date"),
        parse_date("CLM_ADMSN_DT").alias("admit_date"),
        parse_date("NCH_BENE_DSCHRG_DT").alias("discharge_date"),
        nullify_blank("PRVDR_NUM").alias("provider_id"),
        nullify_blank("CLM_DRG_CD").alias("drg_code"),
        nullify_blank("ADMTNG_ICD9_DGNS_CD").alias("admitting_diagnosis_code"),
        parse_double("CLM_PMT_AMT").alias("payment_amount"),
        parse_double("NCH_PRMRY_PYR_CLM_PD_AMT").alias("primary_payer_paid_amount"),
        parse_double("CLM_PASS_THRU_PER_DIEM_AMT").alias("pass_through_per_diem_amount"),
        parse_double("NCH_BENE_IP_DDCTBL_AMT").alias("beneficiary_deductible_amount"),
        parse_double("NCH_BENE_PTA_COINSRNC_LBLTY_AM").alias("beneficiary_coinsurance_amount"),
        parse_double("NCH_BENE_BLOOD_DDCTBL_LBLTY_AM").alias("beneficiary_blood_deductible_amount"),
        parse_int("CLM_UTLZTN_DAY_CNT").alias("claim_days"),
        F.year(parse_date("CLM_THRU_DT")).alias("year"),
        count_true(parse_failure_exprs).alias("_parse_failure_count"),
        F.col("_bronze_source_file").alias("_bronze_source_file"),
        F.col("_bronze_loaded_at_utc").alias("_bronze_loaded_at_utc"),
    )
    return add_processing_metadata(cleaned, BRONZE_TABLES["inpatient_claims"], run_ts)


def build_outpatient_claims(df: DataFrame, run_ts: Column) -> DataFrame:
    from_date = parse_date("CLM_FROM_DT")
    thru_date = parse_date("CLM_THRU_DT")
    hcpcs_columns = [f"HCPCS_CD_{index}" for index in range(1, 46)]
    ed_hcpcs_codes = ["99281", "99282", "99283", "99284", "99285"]
    hcpcs_values = [nullify_blank(column_name) for column_name in hcpcs_columns]
    parse_failure_exprs = [
        int_parse_failed("SEGMENT"),
        date_parse_failed("CLM_FROM_DT"),
        date_parse_failed("CLM_THRU_DT"),
        double_parse_failed("CLM_PMT_AMT"),
        double_parse_failed("NCH_PRMRY_PYR_CLM_PD_AMT"),
        double_parse_failed("NCH_BENE_PTB_DDCTBL_AMT"),
        double_parse_failed("NCH_BENE_PTB_COINSRNC_AMT"),
        double_parse_failed("NCH_BENE_BLOOD_DDCTBL_LBLTY_AM"),
    ]

    cleaned = df.select(
        nullify_blank("CLM_ID").alias("claim_id"),
        nullify_blank("DESYNPUF_ID").alias("bene_id"),
        parse_int("SEGMENT").alias("claim_segment"),
        from_date.alias("claim_from_date"),
        thru_date.alias("claim_thru_date"),
        nullify_blank("PRVDR_NUM").alias("provider_id"),
        nullify_blank("ADMTNG_ICD9_DGNS_CD").alias("admitting_diagnosis_code"),
        F.array_distinct(
            F.filter(
                F.array(*hcpcs_values),
                lambda hcpcs_code: hcpcs_code.isNotNull(),
            )
        ).alias("hcpcs_codes"),
        count_non_null(hcpcs_columns).alias("estimated_line_count"),
        F.exists(F.array(*hcpcs_values), lambda hcpcs_code: hcpcs_code.isin(*ed_hcpcs_codes)).alias(
            "emergency_department_claim_flag"
        ),
        parse_double("CLM_PMT_AMT").alias("payment_amount"),
        parse_double("NCH_PRMRY_PYR_CLM_PD_AMT").alias("primary_payer_paid_amount"),
        parse_double("NCH_BENE_PTB_DDCTBL_AMT").alias("beneficiary_deductible_amount"),
        parse_double("NCH_BENE_PTB_COINSRNC_AMT").alias("beneficiary_coinsurance_amount"),
        parse_double("NCH_BENE_BLOOD_DDCTBL_LBLTY_AM").alias("beneficiary_blood_deductible_amount"),
        F.when(
            from_date.isNotNull() & thru_date.isNotNull(),
            F.datediff(thru_date, from_date) + F.lit(1),
        )
        .otherwise(F.lit(None).cast("int"))
        .alias("claim_days"),
        F.when(from_date.isNotNull() & thru_date.isNotNull() & (from_date <= thru_date), F.lit(True))
        .when(from_date.isNotNull() & thru_date.isNotNull(), F.lit(False))
        .otherwise(F.lit(None).cast("boolean"))
        .alias("claim_dates_valid"),
        F.year(thru_date).alias("year"),
        count_true(parse_failure_exprs).alias("_parse_failure_count"),
        F.col("_bronze_source_file").alias("_bronze_source_file"),
        F.col("_bronze_loaded_at_utc").alias("_bronze_loaded_at_utc"),
    )
    return add_processing_metadata(cleaned, BRONZE_TABLES["outpatient_claims"], run_ts)


def build_carrier_claims(df: DataFrame, run_ts: Column) -> DataFrame:
    provider_columns = [f"PRF_PHYSN_NPI_{index}" for index in range(1, 14)]
    hcpcs_columns = [f"HCPCS_CD_{index}" for index in range(1, 14)]
    payment_columns = [f"LINE_NCH_PMT_AMT_{index}" for index in range(1, 14)]
    allowed_amount_columns = [f"LINE_ALOWD_CHRG_AMT_{index}" for index in range(1, 14)]
    parse_failure_exprs = [
        date_parse_failed("CLM_FROM_DT"),
        date_parse_failed("CLM_THRU_DT"),
        *[double_parse_failed(column_name) for column_name in payment_columns],
        *[double_parse_failed(column_name) for column_name in allowed_amount_columns],
    ]

    cleaned = df.select(
        nullify_blank("CLM_ID").alias("claim_id"),
        nullify_blank("DESYNPUF_ID").alias("bene_id"),
        parse_date("CLM_FROM_DT").alias("claim_from_date"),
        parse_date("CLM_THRU_DT").alias("claim_thru_date"),
        first_non_null(provider_columns).alias("provider_id"),
        F.array_distinct(
            F.filter(
                F.array(*[nullify_blank(column_name) for column_name in provider_columns]),
                lambda provider_id: provider_id.isNotNull(),
            )
        ).alias("provider_ids"),
        F.greatest(count_non_null(hcpcs_columns), count_non_null(payment_columns)).alias("estimated_line_count"),
        sum_columns(payment_columns).alias("payment_amount"),
        sum_columns(allowed_amount_columns).alias("allowed_amount"),
        F.year(parse_date("CLM_THRU_DT")).alias("year"),
        count_true(parse_failure_exprs).alias("_parse_failure_count"),
        F.col("_bronze_source_file").alias("_bronze_source_file"),
        F.col("_bronze_loaded_at_utc").alias("_bronze_loaded_at_utc"),
    )
    return add_processing_metadata(cleaned, BRONZE_TABLES["carrier_claims"], run_ts)


def build_pde(df: DataFrame, run_ts: Column) -> DataFrame:
    service_date = parse_date("SRVC_DT")
    parse_failure_exprs = [
        date_parse_failed("SRVC_DT"),
        double_parse_failed("QTY_DSPNSD_NUM"),
        int_parse_failed("DAYS_SUPLY_NUM"),
        double_parse_failed("PTNT_PAY_AMT"),
        double_parse_failed("TOT_RX_CST_AMT"),
    ]
    cleaned = df.select(
        nullify_blank("PDE_ID").alias("pde_id"),
        nullify_blank("DESYNPUF_ID").alias("bene_id"),
        service_date.alias("drug_fill_date"),
        nullify_blank("PROD_SRVC_ID").alias("product_service_id"),
        parse_double("QTY_DSPNSD_NUM").alias("quantity_dispensed"),
        parse_int("DAYS_SUPLY_NUM").alias("days_supply"),
        parse_double("PTNT_PAY_AMT").alias("patient_pay_amount"),
        parse_double("TOT_RX_CST_AMT").alias("drug_cost"),
        F.year(service_date).alias("year"),
        count_true(parse_failure_exprs).alias("_parse_failure_count"),
        F.col("_bronze_source_file").alias("_bronze_source_file"),
        F.col("_bronze_loaded_at_utc").alias("_bronze_loaded_at_utc"),
    )
    return add_processing_metadata(cleaned, BRONZE_TABLES["prescription_drug_events"], run_ts)


def write_table(df: DataFrame, table_name: str) -> None:
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{SILVER_DATABASE}.{table_name}")
    )


def write_audit_rows(audit_rows: list[dict[str, object]]) -> None:
    if audit_rows:
        (
            build_audit_dataframe(audit_rows)
            .write.format("delta")
            .mode("append")
            .option("mergeSchema", "true")
            .saveAsTable(f"{SILVER_DATABASE}.{AUDIT_TABLE_NAME}")
        )
        audit_rows.clear()


def distinct_key_count(df: DataFrame, columns: list[str]) -> int:
    row = df.agg(
        F.countDistinct(F.struct(*[F.col(column_name) for column_name in columns])).alias("distinct_key_count")
    ).collect()[0]
    return int(row["distinct_key_count"])


def quality_metrics(entity_name: str, df: DataFrame) -> dict[str, int]:
    key_columns = DEDUPE_KEYS[entity_name]
    null_key_condition = F.lit(False)
    for column_name in key_columns:
        null_key_condition = null_key_condition | F.col(column_name).isNull()

    if entity_name == "beneficiary_summary":
        invalid_range_condition = (
            F.col("birth_date").isNotNull()
            & F.col("death_date").isNotNull()
            & (F.col("birth_date") > F.col("death_date"))
        )
        for column_name in [
            "hi_coverage_months",
            "smi_coverage_months",
            "hmo_coverage_months",
            "plan_coverage_months",
        ]:
            invalid_range_condition = invalid_range_condition | (
                (F.col(column_name) < 0) | (F.col(column_name) > 12)
            )
        negative_amount_condition = (
            (F.col("beneficiary_inpatient_reimbursement_amt") < 0)
            | (F.col("beneficiary_outpatient_reimbursement_amt") < 0)
            | (F.col("beneficiary_carrier_reimbursement_amt") < 0)
            | (F.col("beneficiary_inpatient_responsibility_amt") < 0)
            | (F.col("beneficiary_outpatient_responsibility_amt") < 0)
            | (F.col("beneficiary_carrier_responsibility_amt") < 0)
            | (F.col("beneficiary_inpatient_primary_payer_amt") < 0)
            | (F.col("beneficiary_outpatient_primary_payer_amt") < 0)
            | (F.col("beneficiary_carrier_primary_payer_amt") < 0)
        )
    elif entity_name == "prescription_drug_events":
        invalid_range_condition = F.col("days_supply") < 0
        negative_amount_condition = (
            (F.col("quantity_dispensed") < 0)
            | (F.col("patient_pay_amount") < 0)
            | (F.col("drug_cost") < 0)
        )
    elif entity_name == "inpatient_claims":
        invalid_range_condition = (
            (
                F.col("claim_from_date").isNotNull()
                & F.col("claim_thru_date").isNotNull()
                & (F.col("claim_from_date") > F.col("claim_thru_date"))
            )
            | (
                F.col("admit_date").isNotNull()
                & F.col("discharge_date").isNotNull()
                & (F.col("admit_date") > F.col("discharge_date"))
            )
            | (F.col("claim_days") < 0)
        )
        negative_amount_condition = (
            (F.col("payment_amount") < 0)
            | (F.col("primary_payer_paid_amount") < 0)
            | (F.col("pass_through_per_diem_amount") < 0)
            | (F.col("beneficiary_deductible_amount") < 0)
            | (F.col("beneficiary_coinsurance_amount") < 0)
            | (F.col("beneficiary_blood_deductible_amount") < 0)
        )
    else:
        invalid_range_condition = (
            F.col("claim_from_date").isNotNull()
            & F.col("claim_thru_date").isNotNull()
            & (F.col("claim_from_date") > F.col("claim_thru_date"))
        )
        if entity_name == "outpatient_claims":
            negative_amount_condition = (
                (F.col("payment_amount") < 0)
                | (F.col("primary_payer_paid_amount") < 0)
                | (F.col("beneficiary_deductible_amount") < 0)
                | (F.col("beneficiary_coinsurance_amount") < 0)
                | (F.col("beneficiary_blood_deductible_amount") < 0)
            )
        else:
            negative_amount_condition = (F.col("payment_amount") < 0) | (F.col("allowed_amount") < 0)

    row = df.agg(
        F.sum(F.when(F.col("year").isNull(), 1).otherwise(0)).alias("null_year_count"),
        F.sum(F.when(null_key_condition, 1).otherwise(0)).alias("null_key_count"),
        F.sum(F.when(invalid_range_condition, 1).otherwise(0)).alias("invalid_range_count"),
        F.sum(F.when(negative_amount_condition, 1).otherwise(0)).alias("negative_amount_count"),
        F.sum(F.coalesce(F.col("_parse_failure_count"), F.lit(0))).alias("parse_failure_count"),
    ).collect()[0]
    duplicate_key_count = df.count() - distinct_key_count(df, key_columns)

    return {
        "null_year_count": int(row["null_year_count"] or 0),
        "null_key_count": int(row["null_key_count"] or 0),
        "invalid_range_count": int(row["invalid_range_count"] or 0),
        "negative_amount_count": int(row["negative_amount_count"] or 0),
        "parse_failure_count": int(row["parse_failure_count"] or 0),
        "duplicate_key_count": int(duplicate_key_count),
    }


def enforce_quality_gate(entity_name: str, quality: dict[str, int]) -> None:
    fatal_metrics = {
        "null_year_count",
        "null_key_count",
        "invalid_range_count",
        "parse_failure_count",
    }
    failed_metrics = {
        metric_name: count
        for metric_name, count in quality.items()
        if metric_name in fatal_metrics and count > 0
    }
    if failed_metrics:
        raise ValueError(f"Refusing to publish {entity_name}: quality gate failed {failed_metrics}")


def empty_quality_metrics() -> dict[str, int]:
    return {
        "null_year_count": 0,
        "null_key_count": 0,
        "invalid_range_count": 0,
        "negative_amount_count": 0,
        "parse_failure_count": 0,
        "duplicate_key_count": 0,
    }


def build_audit_dataframe(audit_rows: list[dict[str, object]]) -> DataFrame:
    schema = T.StructType(
        [
            T.StructField("source_table", T.StringType(), False),
            T.StructField("target_table", T.StringType(), False),
            T.StructField("raw_row_count", T.LongType(), False),
            T.StructField("silver_row_count", T.LongType(), False),
            T.StructField("rows_removed", T.LongType(), True),
            T.StructField("null_year_count", T.LongType(), False),
            T.StructField("null_key_count", T.LongType(), False),
            T.StructField("invalid_range_count", T.LongType(), False),
            T.StructField("negative_amount_count", T.LongType(), False),
            T.StructField("parse_failure_count", T.LongType(), False),
            T.StructField("duplicate_key_count", T.LongType(), False),
            T.StructField("status", T.StringType(), False),
            T.StructField("error_message", T.StringType(), True),
            T.StructField("run_id", T.StringType(), False),
            T.StructField("processed_at_utc", T.TimestampType(), True),
        ]
    )
    return spark.createDataFrame(audit_rows, schema=schema)


def append_audit_row(
    audit_rows: list[dict[str, object]],
    source_table: str,
    target_table: str,
    raw_row_count: int,
    silver_row_count: int,
    quality: dict[str, int],
    run_processed_at_utc: datetime,
    run_id: str,
    status: str,
    error_message: str | None = None,
) -> None:
    rows_removed = raw_row_count - silver_row_count if status == "published" else None
    audit_rows.append(
        {
            "source_table": f"{BRONZE_DATABASE}.{source_table}",
            "target_table": f"{SILVER_DATABASE}.{target_table}",
            "raw_row_count": raw_row_count,
            "silver_row_count": silver_row_count,
            "rows_removed": rows_removed,
            "null_year_count": quality["null_year_count"],
            "null_key_count": quality["null_key_count"],
            "invalid_range_count": quality["invalid_range_count"],
            "negative_amount_count": quality["negative_amount_count"],
            "parse_failure_count": quality["parse_failure_count"],
            "duplicate_key_count": quality["duplicate_key_count"],
            "status": status,
            "error_message": error_message,
            "run_id": run_id,
            "processed_at_utc": run_processed_at_utc,
        }
    )


def main() -> None:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {SILVER_DATABASE}")
    run_id = str(uuid4())
    run_processed_at_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    run_ts = F.lit(run_processed_at_utc).cast("timestamp")

    builders = {
        "beneficiary_summary": build_beneficiaries,
        "inpatient_claims": build_inpatient_claims,
        "outpatient_claims": build_outpatient_claims,
        "carrier_claims": build_carrier_claims,
        "prescription_drug_events": build_pde,
    }

    audit_rows: list[dict[str, object]] = []

    for entity_name, builder in builders.items():
        source_table = BRONZE_TABLES[entity_name]
        target_table = SILVER_TABLES[entity_name]
        raw_row_count = 0
        silver_row_count = 0
        quality = None
        audit_recorded = False
        try:
            bronze_df = read_bronze(source_table)
            raw_row_count = bronze_df.count()
            silver_raw_df = builder(bronze_df, run_ts).dropna(subset=DEDUPE_KEYS[entity_name])
            quality = quality_metrics(entity_name, silver_raw_df)
            enforce_quality_gate(entity_name, quality)

            silver_df = deduplicate_latest(silver_raw_df, DEDUPE_KEYS[entity_name])
            silver_row_count = silver_df.count()
            write_table(silver_df, target_table)
            append_audit_row(
                audit_rows,
                source_table,
                target_table,
                raw_row_count,
                silver_row_count,
                quality,
                run_processed_at_utc,
                run_id,
                "published",
            )
            audit_recorded = True
            print(
                f"{entity_name}: source={BRONZE_DATABASE}.{source_table} "
                f"target={SILVER_DATABASE}.{target_table} raw_row_count={raw_row_count} "
                f"silver_row_count={silver_row_count} null_year_count={quality['null_year_count']} "
                f"null_key_count={quality['null_key_count']} invalid_range_count={quality['invalid_range_count']} "
                f"negative_amount_count={quality['negative_amount_count']} "
                f"parse_failure_count={quality['parse_failure_count']} "
                f"duplicate_key_count={quality['duplicate_key_count']}"
            )
        except Exception as exc:
            if not audit_recorded:
                append_audit_row(
                    audit_rows,
                    source_table,
                    target_table,
                    raw_row_count,
                    silver_row_count,
                    quality or empty_quality_metrics(),
                    run_processed_at_utc,
                    run_id,
                    "failed",
                    str(exc),
                )
                write_audit_rows(audit_rows)
            raise RuntimeError(f"Failed processing silver entity {entity_name}") from exc

    write_audit_rows(audit_rows)
    print(f"silver audit summary written to {SILVER_DATABASE}.{AUDIT_TABLE_NAME}")


# COMMAND ----------

main()
