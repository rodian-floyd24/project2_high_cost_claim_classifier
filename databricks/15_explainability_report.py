# Databricks notebook source
"""Create a lightweight explainability artifact from model prediction scores.

This notebook is intentionally conservative: reason-code methodology lives in
`docs/explainability_methodology.md`, while production-grade SHAP/PDP artifacts
can be added after a champion model is locked.
"""

from __future__ import annotations

import os

from pyspark.sql import functions as F

from databricks.run_selection_utils import filter_to_selected_test_rows, selected_test_runs


MODEL_DATABASE = os.environ.get("MODEL_DATABASE", os.environ.get("GOLD_DATABASE", "default"))
PREDICTION_SCORE_TABLE = "model_prediction_scores"
EXPLAINABILITY_AUDIT_TABLE = "model_explainability_audit"


def main() -> None:
    source = f"{MODEL_DATABASE}.{PREDICTION_SCORE_TABLE}"
    scores = spark.table(source)
    latest = filter_to_selected_test_rows(
        scores,
        selected_test_runs(spark, MODEL_DATABASE),
        source,
    )
    summary = (
        latest.groupBy("model_name", "split_name", "run_id")
        .agg(
            F.count("*").alias("row_count"),
            F.avg("predicted_probability").alias("mean_predicted_probability"),
            F.avg("label").alias("observed_positive_rate"),
            F.max("predicted_probability").alias("max_predicted_probability"),
        )
        .withColumn("reason_code_version", F.lit("v1"))
        .withColumn("methodology", F.lit("rule-based actuarial reason codes plus global score distribution checks"))
        .withColumn("processed_at_utc", F.current_timestamp())
    )
    summary.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
        f"{MODEL_DATABASE}.{EXPLAINABILITY_AUDIT_TABLE}"
    )
    display(summary)


# COMMAND ----------

main()
