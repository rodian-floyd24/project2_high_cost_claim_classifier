# Databricks notebook source
# MAGIC %md
# MAGIC # Target Definition Sensitivity
# MAGIC
# MAGIC Compares prospective high-cost target definitions by target year so label drift is visible before model training.

# COMMAND ----------

from __future__ import annotations

import os
from datetime import datetime, timezone

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


GOLD_DATABASE = os.environ.get("GOLD_DATABASE", "default")
MODEL_DATABASE = os.environ.get("MODEL_DATABASE", GOLD_DATABASE)
GOLD_TABLE_NAME = "gold_beneficiary_year_features"
TARGET_SENSITIVITY_TABLE = "target_definition_sensitivity"
FIXED_THRESHOLD_QUANTILE = float(os.environ.get("FIXED_THRESHOLD_QUANTILE", "0.9"))


TARGET_DEFINITIONS = [
    {
        "target_definition": "fixed_train_threshold_top_decile",
        "quantile": 0.9,
        "within_year": False,
        "full_year_only": False,
    },
    {
        "target_definition": "within_target_year_top_decile",
        "quantile": 0.9,
        "within_year": True,
        "full_year_only": False,
    },
    {
        "target_definition": "within_target_year_top_decile_full_year_enrolled",
        "quantile": 0.9,
        "within_year": True,
        "full_year_only": True,
    },
    {
        "target_definition": "within_target_year_top_5_percent",
        "quantile": 0.95,
        "within_year": True,
        "full_year_only": False,
    },
]


def read_gold() -> DataFrame:
    return spark.table(f"{GOLD_DATABASE}.{GOLD_TABLE_NAME}")


def build_modeling_frame(df: DataFrame) -> DataFrame:
    current_year = df.alias("current")
    next_year = df.alias("next")
    return (
        current_year.join(
            next_year,
            (F.col("current.bene_id") == F.col("next.bene_id"))
            & (F.col("current.year") + F.lit(1) == F.col("next.year")),
            "inner",
        )
        .select(
            F.col("current.bene_id"),
            F.col("current.year").alias("feature_year"),
            F.col("next.year").alias("target_year"),
            F.col("next.annual_claim_cost").alias("target_annual_claim_cost"),
            F.col("next.enrollment_months_count").alias("target_enrollment_months_count"),
        )
    )


def historical_train_pool(df: DataFrame) -> DataFrame:
    latest_target_year = df.agg(F.max("target_year").alias("latest_target_year")).collect()[0]["latest_target_year"]
    return df.filter(F.col("target_year") < F.lit(latest_target_year))


def threshold_frame(df: DataFrame, definition: dict[str, object], fixed_threshold: float) -> DataFrame:
    quantile = float(definition["quantile"])
    full_year_only = bool(definition["full_year_only"])
    threshold_source = df.filter(F.col("target_enrollment_months_count") == 12) if full_year_only else df

    if bool(definition["within_year"]):
        return threshold_source.groupBy("target_year").agg(
            F.expr(f"percentile_approx(target_annual_claim_cost, {quantile})").alias("target_threshold")
        )

    return df.select("target_year").distinct().withColumn("target_threshold", F.lit(float(fixed_threshold)))


def sensitivity_rows(df: DataFrame, definition: dict[str, object], fixed_threshold: float) -> DataFrame:
    definition_name = str(definition["target_definition"])
    full_year_only = bool(definition["full_year_only"])
    thresholds = threshold_frame(df, definition, fixed_threshold)
    labeled = (
        df.join(thresholds, "target_year", "left")
        .withColumn(
            "label",
            (
                (F.col("target_annual_claim_cost") > F.col("target_threshold"))
                & ((F.col("target_enrollment_months_count") == 12) if full_year_only else F.lit(True))
            ).cast("int"),
        )
        .withColumn("eligible_for_positive", ((F.col("target_enrollment_months_count") == 12) if full_year_only else F.lit(True)).cast("int"))
    )

    return (
        labeled.groupBy("target_year")
        .agg(
            F.count("*").alias("row_count"),
            F.sum("eligible_for_positive").cast("long").alias("eligible_positive_denominator"),
            F.avg("label").alias("positive_rate_all_rows"),
            safe_avg_when(F.col("eligible_for_positive") == 1, F.col("label")).alias("positive_rate_eligible_rows"),
            F.sum("label").cast("long").alias("positive_count"),
            F.first("target_threshold", ignorenulls=True).alias("target_threshold"),
            F.avg("target_annual_claim_cost").alias("mean_target_annual_claim_cost"),
            F.expr("percentile_approx(target_annual_claim_cost, 0.5)").alias("median_target_annual_claim_cost"),
            F.avg("target_enrollment_months_count").alias("mean_target_enrollment_months"),
        )
        .withColumn("target_definition", F.lit(definition_name))
        .withColumn("target_quantile", F.lit(float(definition["quantile"])))
        .withColumn("within_year_threshold", F.lit(bool(definition["within_year"])))
        .withColumn("requires_full_year_enrollment", F.lit(full_year_only))
    )


def safe_avg_when(condition, value):
    return F.avg(F.when(condition, value).otherwise(F.lit(None)))


def main() -> None:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {MODEL_DATABASE}")
    modeling_df = build_modeling_frame(read_gold())
    train_pool = historical_train_pool(modeling_df)
    fixed_threshold = train_pool.approxQuantile("target_annual_claim_cost", [FIXED_THRESHOLD_QUANTILE], 0.001)[0]
    if fixed_threshold is None:
        raise ValueError("Cannot compute fixed threshold because the historical training pool is empty.")

    sensitivity_df = None
    for definition in TARGET_DEFINITIONS:
        definition_df = sensitivity_rows(modeling_df, definition, float(fixed_threshold))
        sensitivity_df = definition_df if sensitivity_df is None else sensitivity_df.unionByName(definition_df)

    sensitivity_df = sensitivity_df.withColumn("fixed_train_threshold", F.lit(float(fixed_threshold))).withColumn(
        "processed_at_utc", F.lit(datetime.now(timezone.utc).replace(tzinfo=None))
    )

    (
        sensitivity_df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{MODEL_DATABASE}.{TARGET_SENSITIVITY_TABLE}")
    )

    print(
        f"target sensitivity table written to {MODEL_DATABASE}.{TARGET_SENSITIVITY_TABLE}; "
        f"fixed historical-train threshold={fixed_threshold:.2f}"
    )
    display(sensitivity_df.orderBy("target_definition", "target_year"))


# COMMAND ----------

main()
