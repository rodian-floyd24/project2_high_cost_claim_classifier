# Databricks notebook source
"""Model monitoring scaffold for score, calibration, and top-k drift checks."""

from __future__ import annotations

import os

from pyspark.sql import functions as F

from databricks.run_selection_utils import filter_to_selected_test_rows, selected_test_runs


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
    current_scores = filter_to_selected_test_rows(
        scores,
        selected_test_runs(spark, MODEL_DATABASE),
        f"{MODEL_DATABASE}.{PREDICTION_SCORE_TABLE}",
    )
    summary = (
        current_scores.groupBy("model_name", "split_name", "run_id")
        .agg(
            F.count("*").alias("row_count"),
            F.avg("predicted_probability").alias("mean_prediction"),
            F.avg("label").alias("observed_rate"),
            F.max("processed_at_utc").alias("latest_score_processed_at_utc"),
        )
        .withColumn("absolute_calibration_gap", F.abs(F.col("mean_prediction") - F.col("observed_rate")))
        .withColumn("monitoring_scope", F.lit("current_selected_run"))
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
