# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Pipeline Consistency Check
# MAGIC
# MAGIC Verifies that EDA, training, and evaluation are anchored to the same gold feature table contract.

# COMMAND ----------

from __future__ import annotations

import os

from pyspark.sql import functions as F
from pyspark.sql import types as T

from shared.feature_contract import FEATURE_VERSION


GOLD_DATABASE = os.environ.get("GOLD_DATABASE", "default")
MODEL_DATABASE = os.environ.get("MODEL_DATABASE", GOLD_DATABASE)
GOLD_TABLE_NAME = "gold_beneficiary_year_features"
CONSISTENCY_AUDIT_TABLE = "gold_pipeline_consistency_audit"
EXPECTED_GOLD_FEATURE_VERSION = FEATURE_VERSION
SHARED_SPLIT_VERSION = "xxhash64_bene_id_mod_100_v2_beneficiary_hash_holdout"

TRAINING_AUDIT_TABLES = {
    "logistic_regression": "logreg_training_audit",
    "random_forest": "tree_training_audit",
    "gradient_boosting": "boosted_tree_training_audit",
    "xgboost": "xgboost_training_audit",
}
OPTIONAL_STATISTICAL_AUDIT_TABLES = {
    "logistic_regression_selected_bic": "logreg_variable_selection_audit",
}

REQUIRED_GOLD_COLUMNS = [
    "bene_id",
    "year",
    "annual_claim_cost",
    "chronic_condition_count",
    "chronic_burden_band",
    "claims_per_enrollment_month",
    "claims_per_month_chronic_count_interaction",
    "providers_per_month_chronic_count_interaction",
    "chronic_condition_count_squared",
    "inpatient_claim_count_log1p",
    "outpatient_claim_count_log1p",
    "carrier_claim_count_log1p",
    "pde_claim_count_log1p",
    "total_claim_count_log1p",
    "unique_provider_count_log1p",
]


def table_exists(table_name: str) -> bool:
    try:
        spark.table(table_name).limit(1).count()
        return True
    except Exception:
        return False


def get_table_property(table_name: str, property_name: str) -> str | None:
    try:
        rows = spark.sql(f"SHOW TBLPROPERTIES {table_name}('{property_name}')").collect()
        if not rows:
            return None
        value = rows[0].asDict().get("value")
        return None if value is None else str(value)
    except Exception:
        return None


def latest_run_rows(table_name: str):
    df = spark.table(table_name)
    latest_ts = df.agg(F.max("processed_at_utc").alias("processed_at_utc")).collect()[0]["processed_at_utc"]
    return df.filter(F.col("processed_at_utc") == F.lit(latest_ts))


def add_result(rows: list[dict[str, object]], check_name: str, status: str, details: str) -> None:
    rows.append({"check_name": check_name, "status": status, "details": details})


def main() -> None:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {MODEL_DATABASE}")
    rows: list[dict[str, object]] = []
    gold_table = f"{GOLD_DATABASE}.{GOLD_TABLE_NAME}"

    if not table_exists(gold_table):
        add_result(rows, "gold_table_exists", "fail", f"{gold_table} is not readable")
    else:
        gold_df = spark.table(gold_table)
        add_result(rows, "gold_table_exists", "pass", gold_table)

        missing_columns = sorted(set(REQUIRED_GOLD_COLUMNS) - set(gold_df.columns))
        add_result(
            rows,
            "gold_required_columns",
            "fail" if missing_columns else "pass",
            "missing=" + ",".join(missing_columns) if missing_columns else "all required columns present",
        )

        feature_version = get_table_property(gold_table, "gold_feature_version")
        add_result(
            rows,
            "gold_feature_version",
            "pass" if feature_version == EXPECTED_GOLD_FEATURE_VERSION else "fail",
            f"actual={feature_version}; expected={EXPECTED_GOLD_FEATURE_VERSION}",
        )
        for property_name, expected_value in {
            "grain": "beneficiary_year",
            "primary_key": "bene_id,year",
            "source_system": "cms_desynpuf",
        }.items():
            actual_value = get_table_property(gold_table, property_name)
            add_result(
                rows,
                f"gold_property_{property_name}",
                "pass" if actual_value == expected_value else "fail",
                f"actual={actual_value}; expected={expected_value}",
            )

        duplicate_count = gold_df.groupBy("bene_id", "year").count().filter(F.col("count") > 1).limit(1).count()
        add_result(
            rows,
            "gold_unique_bene_year",
            "fail" if duplicate_count else "pass",
            "duplicate bene_id/year rows exist" if duplicate_count else "bene_id/year is unique",
        )

        chronic_stats = gold_df.agg(
            F.min("chronic_condition_count").alias("min_chronic"),
            F.max("chronic_condition_count").alias("max_chronic"),
            F.countDistinct("chronic_condition_count").alias("distinct_chronic"),
        ).collect()[0]
        chronic_ok = (
            chronic_stats["min_chronic"] >= 0
            and chronic_stats["max_chronic"] <= 11
            and chronic_stats["distinct_chronic"] > 1
        )
        add_result(
            rows,
            "gold_chronic_count_populated",
            "pass" if chronic_ok else "fail",
            (
                f"min={chronic_stats['min_chronic']}; max={chronic_stats['max_chronic']}; "
                f"distinct={chronic_stats['distinct_chronic']}"
            ),
        )

    for model_name, audit_table_name in TRAINING_AUDIT_TABLES.items():
        audit_table = f"{MODEL_DATABASE}.{audit_table_name}"
        if not table_exists(audit_table):
            add_result(rows, f"{model_name}_audit_exists", "fail", f"{audit_table} is not readable")
            continue

        latest = latest_run_rows(audit_table)
        split_names = {row["split_name"] for row in latest.select("split_name").distinct().collect()}
        row_counts = {row["split_name"]: row["row_count"] for row in latest.select("split_name", "row_count").collect()}
        split_ok = {"train", "validation", "test"}.issubset(split_names)
        add_result(
            rows,
            f"{model_name}_latest_splits",
            "pass" if split_ok else "fail",
            f"splits={sorted(split_names)}; row_counts={row_counts}",
        )

        if "shared_split_version" in latest.columns:
            split_versions = {row["shared_split_version"] for row in latest.select("shared_split_version").distinct().collect()}
            add_result(
                rows,
                f"{model_name}_shared_split_version",
                "pass" if split_versions == {SHARED_SPLIT_VERSION} else "fail",
                f"versions={sorted(split_versions)}",
            )

    for model_name, audit_table_name in OPTIONAL_STATISTICAL_AUDIT_TABLES.items():
        audit_table = f"{MODEL_DATABASE}.{audit_table_name}"
        if not table_exists(audit_table):
            add_result(rows, f"{model_name}_audit_exists", "warn", f"{audit_table} has not been run yet")
            continue
        latest = latest_run_rows(audit_table)
        split_names = {row["split_name"] for row in latest.select("split_name").distinct().collect()}
        split_ok = {"train", "validation", "test"}.issubset(split_names)
        add_result(
            rows,
            f"{model_name}_latest_splits",
            "pass" if split_ok else "fail",
            f"splits={sorted(split_names)}",
        )

    output_schema = T.StructType(
        [
            T.StructField("check_name", T.StringType(), False),
            T.StructField("status", T.StringType(), False),
            T.StructField("details", T.StringType(), False),
        ]
    )
    output_df = spark.createDataFrame(rows, schema=output_schema).withColumn("processed_at_utc", F.current_timestamp())
    (
        output_df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{MODEL_DATABASE}.{CONSISTENCY_AUDIT_TABLE}")
    )

    failures = output_df.filter(F.col("status") == F.lit("fail")).count()
    display(output_df.orderBy("check_name"))
    if failures:
        raise ValueError(f"Gold pipeline consistency check failed with {failures} failing checks.")

    print(f"gold pipeline consistency audit written to {MODEL_DATABASE}.{CONSISTENCY_AUDIT_TABLE}")


# COMMAND ----------

main()
