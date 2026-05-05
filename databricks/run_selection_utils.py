from __future__ import annotations

from pyspark.sql import functions as F


COMPARISON_TABLE_NAME = "model_comparison_summary"
FINAL_EVALUATION_SPLIT = "test"


def validate_one_run_per_model(df, context: str) -> None:
    if df.count() == 0:
        raise ValueError(f"{context} did not return any selected model runs.")

    invalid = (
        df.groupBy("model_name")
        .agg(
            F.count("*").alias("row_count"),
            F.countDistinct("run_id").alias("distinct_run_count"),
        )
        .filter((F.col("row_count") != F.lit(1)) | (F.col("distinct_run_count") != F.lit(1)))
    )
    invalid_rows = invalid.collect()
    if invalid_rows:
        details = ", ".join(str(row.asDict()) for row in invalid_rows)
        raise ValueError(f"{context} must contain exactly one run_id per model; invalid rows: {details}")


def selected_test_runs(spark, model_database: str, comparison_table_name: str = COMPARISON_TABLE_NAME):
    comparison_df = spark.table(f"{model_database}.{comparison_table_name}")
    selected = comparison_df.filter(F.col("split_name") == F.lit(FINAL_EVALUATION_SPLIT)).select(
        "model_name",
        "run_id",
        F.col("processed_at_utc").alias("selected_run_processed_at_utc"),
    )
    validate_one_run_per_model(selected, f"{model_database}.{comparison_table_name}")
    return selected


def filter_to_selected_test_rows(source_df, selected_runs, context: str):
    selected_keys = selected_runs.select("model_name", "run_id")
    selected_rows = source_df.filter(F.col("split_name") == F.lit(FINAL_EVALUATION_SPLIT)).join(
        selected_keys,
        ["model_name", "run_id"],
        "inner",
    )

    missing = (
        selected_keys.join(
            selected_rows.groupBy("model_name", "run_id").agg(F.count("*").alias("selected_row_count")),
            ["model_name", "run_id"],
            "left",
        )
        .filter(F.col("selected_row_count").isNull() | (F.col("selected_row_count") == F.lit(0)))
        .collect()
    )
    if missing:
        details = ", ".join(str(row.asDict()) for row in missing)
        raise ValueError(f"{context} is missing rows for selected model run_ids: {details}")

    return selected_rows
