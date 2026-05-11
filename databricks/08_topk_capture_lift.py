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

# MAGIC %run ./run_selection_utils

# COMMAND ----------

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import functions as F


MODEL_DATABASE = os.environ.get("MODEL_DATABASE", "default")
CURVE_TABLE_NAME = "model_topk_curve_points"
SHARED_SPLIT_VERSION = "xxhash64_bene_id_mod_100_v2_beneficiary_hash_holdout"


def selected_curve_rows():
    curve_df = spark.table(f"{MODEL_DATABASE}.{CURVE_TABLE_NAME}")
    selected_runs = selected_test_runs(spark, MODEL_DATABASE)
    return filter_to_selected_test_rows(
        curve_df,
        selected_runs,
        f"{MODEL_DATABASE}.{CURVE_TABLE_NAME}",
    )


def main() -> None:
    curve_pdf = selected_curve_rows().toPandas().sort_values(["model_name", "selected_fraction"])
    models = list(curve_pdf["model_name"].drop_duplicates())

    plt.figure(figsize=(10, 6))
    for model_name in models:
        model_df = curve_pdf[curve_pdf["model_name"] == model_name]
        plt.plot(model_df["selected_fraction"], model_df["capture_rate"], marker="o", label=model_name)
    plt.title(f"Business Ranking Diagnostic: Top-K Capture Curve (Test Set, {SHARED_SPLIT_VERSION})")
    plt.xlabel("Selected fraction")
    plt.ylabel("Capture rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 6))
    for model_name in models:
        model_df = curve_pdf[curve_pdf["model_name"] == model_name]
        plt.plot(model_df["selected_fraction"], model_df["lift"], marker="o", label=model_name)
    plt.title(f"Business Ranking Diagnostic: Top-K Lift Curve (Test Set, {SHARED_SPLIT_VERSION})")
    plt.xlabel("Selected fraction")
    plt.ylabel("Lift")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    display(curve_pdf)


# COMMAND ----------

main()
