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
from pyspark.sql.window import Window


MODEL_DATABASE = os.environ.get("MODEL_DATABASE", "default")
COMPARISON_TABLE_NAME = "model_comparison_summary"
FINAL_EVALUATION_SPLIT = "test"
SHARED_SPLIT_VERSION = "xxhash64_bene_id_mod_100_v2_beneficiary_hash_holdout"
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
    if test_df.count() == 0:
        raise ValueError(f"{MODEL_DATABASE}.{table_name} has no completed {FINAL_EVALUATION_SPLIT} audit rows")

    run_window = Window.orderBy(F.col("latest_processed_at_utc").desc(), F.col("run_id").desc())
    latest_run_id = (
        test_df.groupBy("run_id")
        .agg(F.max("processed_at_utc").alias("latest_processed_at_utc"))
        .withColumn("run_rank", F.row_number().over(run_window))
        .filter(F.col("run_rank") == F.lit(1))
        .select("run_id")
        .collect()[0]["run_id"]
    )
    selected_df = test_df.filter(F.col("run_id") == F.lit(latest_run_id))
    if selected_df.count() != 1:
        raise ValueError(
            f"{MODEL_DATABASE}.{table_name} must have exactly one {FINAL_EVALUATION_SPLIT} row "
            f"for selected run_id={latest_run_id}"
        )
    return (
        selected_df
        .withColumn("model_name", F.lit(model_name))
        .withColumn("model_group", F.lit(MODEL_GROUPS[model_name]))
        .withColumn("shared_split_version", F.lit(SHARED_SPLIT_VERSION))
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
            "shared_split_version",
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
        f"under split version {SHARED_SPLIT_VERSION} "
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
