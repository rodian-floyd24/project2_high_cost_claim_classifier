# Databricks notebook source
"""Model monitoring scaffold for score, calibration, and top-k drift checks."""

from __future__ import annotations

import os

from pyspark.sql import functions as F


MODEL_DATABASE = os.environ.get("MODEL_DATABASE", os.environ.get("GOLD_DATABASE", "default"))
PREDICTION_SCORE_TABLE = "model_prediction_scores"
MONITORING_TABLE = "model_monitoring_summary"


def status_from_gap(gap: float) -> str:
    if gap >= 0.05:
        return "review_required"
    if gap >= 0.02:
        return "warning"
    return "acceptable"


def main() -> None:
    scores = spark.table(f"{MODEL_DATABASE}.{PREDICTION_SCORE_TABLE}")
    summary = (
        scores.groupBy("model_name", "split_name")
        .agg(
            F.count("*").alias("row_count"),
            F.avg("predicted_probability").alias("mean_prediction"),
            F.avg("label").alias("observed_rate"),
            F.min("processed_at_utc").alias("first_score_processed_at_utc"),
            F.max("processed_at_utc").alias("latest_score_processed_at_utc"),
        )
        .withColumn("absolute_calibration_gap", F.abs(F.col("mean_prediction") - F.col("observed_rate")))
    )
    status_udf = F.udf(status_from_gap, "string")
    summary = summary.withColumn("calibration_status", status_udf(F.col("absolute_calibration_gap"))).withColumn(
        "monitoring_processed_at_utc", F.current_timestamp()
    )
    summary.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        f"{MODEL_DATABASE}.{MONITORING_TABLE}"
    )
    display(summary)


# COMMAND ----------

main()
