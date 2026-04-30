# Databricks notebook source
# MAGIC %md
# MAGIC # Model Comparison
# MAGIC
# MAGIC Builds a side-by-side held-out test comparison table for the latest logistic, random-forest, gradient-boosting, and XGBoost runs.
# MAGIC
# MAGIC The primary comparison orders models on held-out test ranking quality: PR-AUC first, then top-k capture,
# MAGIC then ROC-AUC. Thresholded metrics are retained as secondary operational diagnostics.

# COMMAND ----------

from __future__ import annotations

import os

from pyspark.sql import functions as F


MODEL_DATABASE = os.environ.get("MODEL_DATABASE", "default")
COMPARISON_TABLE_NAME = "model_comparison_summary"
FINAL_EVALUATION_SPLIT = "test"
MODEL_GROUPS = {
    "logistic_regression": "ISLP core",
    "random_forest": "ISLP core",
    "gradient_boosting": "ISLP core",
    "xgboost": "modern extension",
}


def latest_run(table_name: str, model_name: str):
    df = spark.table(f"{MODEL_DATABASE}.{table_name}")
    threshold_columns = [
        F.col(column_name)
        for column_name in ["decision_threshold_from_tuning_split", "decision_threshold_validation_tuned"]
        if column_name in df.columns
    ]
    if not threshold_columns:
        raise ValueError(f"{MODEL_DATABASE}.{table_name} is missing a decision-threshold audit column")

    test_df = df.filter(F.col("split_name") == F.lit(FINAL_EVALUATION_SPLIT))
    latest_ts = test_df.agg(F.max("processed_at_utc").alias("processed_at_utc")).collect()[0]["processed_at_utc"]
    return (
        test_df.filter(F.col("processed_at_utc") == F.lit(latest_ts))
        .withColumn("model_name", F.lit(model_name))
        .withColumn("model_group", F.lit(MODEL_GROUPS[model_name]))
        .withColumn("decision_threshold_from_tuning_split", F.coalesce(*threshold_columns))
        .withColumn("test_misclassification_error", 1 - F.col("accuracy"))
        .select(
            "model_name",
            "model_group",
            "run_id",
            "split_name",
            "row_count",
            "positive_rate",
            "accuracy",
            "test_misclassification_error",
            "precision",
            "recall",
            "area_under_roc",
            "area_under_pr",
            *([ "brier_score" ] if "brier_score" in test_df.columns else []),
            "top_5_capture_rate",
            "top_5_lift",
            "top_10_capture_rate",
            "top_10_lift",
            "high_cost_threshold_train_only",
            "decision_threshold_from_tuning_split",
            "processed_at_utc",
        )
    )


def main() -> None:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {MODEL_DATABASE}")

    comparison_df = latest_run("logreg_training_audit", "logistic_regression")
    comparison_df = comparison_df.unionByName(latest_run("tree_training_audit", "random_forest"))
    comparison_df = comparison_df.unionByName(latest_run("boosted_tree_training_audit", "gradient_boosting"))
    comparison_df = comparison_df.unionByName(latest_run("xgboost_training_audit", "xgboost"))

    (
        comparison_df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{MODEL_DATABASE}.{COMPARISON_TABLE_NAME}")
    )

    print(
        f"comparison table written to {MODEL_DATABASE}.{COMPARISON_TABLE_NAME}; "
        f"primary model comparison is restricted to the held-out {FINAL_EVALUATION_SPLIT} split "
        "and ordered by PR-AUC, top-5 capture, top-10 capture, ROC-AUC, then calibration"
    )
    display(
        comparison_df.orderBy(
            F.col("area_under_pr").desc(),
            F.col("top_5_capture_rate").desc(),
            F.col("top_10_capture_rate").desc(),
            F.col("area_under_roc").desc(),
            F.col("brier_score").asc_nulls_last(),
            F.col("test_misclassification_error").asc(),
        )
    )


# COMMAND ----------

main()
