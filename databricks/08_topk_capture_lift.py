# Databricks notebook source
# MAGIC %md
# MAGIC # Top-K Capture And Lift Plots
# MAGIC
# MAGIC Plots test-set capture and lift curves for the latest model runs.
# MAGIC
# MAGIC These curves are business decision-support diagnostics for ranking high-cost claim risk. They supplement the
# MAGIC held-out test-error comparison in `07_model_comparison.py`; they are not the primary ISLP-style model
# MAGIC assessment metric.

# COMMAND ----------

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import functions as F


MODEL_DATABASE = os.environ.get("MODEL_DATABASE", "default")
CURVE_TABLE_NAME = "model_topk_curve_points"


def latest_curve_rows(model_name: str):
    df = spark.table(f"{MODEL_DATABASE}.{CURVE_TABLE_NAME}")
    test_curve_df = df.filter((F.col("model_name") == model_name) & (F.col("split_name") == "test"))
    latest_ts = test_curve_df.agg(F.max("processed_at_utc").alias("processed_at_utc")).collect()[0][
        "processed_at_utc"
    ]
    return test_curve_df.filter(F.col("processed_at_utc") == F.lit(latest_ts))


def main() -> None:
    models = ["logistic_regression", "random_forest", "gradient_boosting", "xgboost"]
    curve_pdf = pd.concat(
        [latest_curve_rows(model_name).toPandas() for model_name in models],
        ignore_index=True,
    ).sort_values(["model_name", "selected_fraction"])

    plt.figure(figsize=(10, 6))
    for model_name in models:
        model_df = curve_pdf[curve_pdf["model_name"] == model_name]
        plt.plot(model_df["selected_fraction"], model_df["capture_rate"], marker="o", label=model_name)
    plt.title("Business Ranking Diagnostic: Top-K Capture Curve (Test Set)")
    plt.xlabel("Selected fraction")
    plt.ylabel("Capture rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 6))
    for model_name in models:
        model_df = curve_pdf[curve_pdf["model_name"] == model_name]
        plt.plot(model_df["selected_fraction"], model_df["lift"], marker="o", label=model_name)
    plt.title("Business Ranking Diagnostic: Top-K Lift Curve (Test Set)")
    plt.xlabel("Selected fraction")
    plt.ylabel("Lift")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    display(curve_pdf)


# COMMAND ----------

main()
