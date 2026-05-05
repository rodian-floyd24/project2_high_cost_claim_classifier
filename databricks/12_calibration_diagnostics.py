# Databricks notebook source
# MAGIC %md
# MAGIC # Calibration Diagnostics
# MAGIC
# MAGIC Builds held-out test calibration diagnostics from the shared per-row prediction table so every model is
# MAGIC evaluated on the same scored test population.

# COMMAND ----------

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql import types as T

from databricks.run_selection_utils import filter_to_selected_test_rows, selected_test_runs


MODEL_DATABASE = os.environ.get("MODEL_DATABASE", "default")
PREDICTION_SCORE_TABLE = "model_prediction_scores"
CALIBRATION_SUMMARY_TABLE = "model_calibration_summary"
CALIBRATION_DECILE_TABLE = "model_calibration_deciles"
RELIABILITY_TABLE = "model_reliability_curve_points"


def selected_test_scores():
    score_df = spark.table(f"{MODEL_DATABASE}.{PREDICTION_SCORE_TABLE}")
    selected_runs = selected_test_runs(spark, MODEL_DATABASE)
    return filter_to_selected_test_rows(
        score_df,
        selected_runs,
        f"{MODEL_DATABASE}.{PREDICTION_SCORE_TABLE}",
    )


def append_decile_columns(pdf: pd.DataFrame) -> pd.DataFrame:
    scored = pdf.copy()
    scored["rank_order"] = scored["predicted_probability"].rank(method="first", ascending=False)
    scored["score_decile"] = (((scored["rank_order"] - 1) * 10) / len(scored)).astype(int).clip(0, 9) + 1
    return scored.drop(columns=["rank_order"])


def calibration_summary_rows(model_name: str, run_id: str, pdf: pd.DataFrame) -> list[dict[str, object]]:
    brier_score = float(((pdf["predicted_probability"] - pdf["label"]) ** 2).mean())
    average_predicted = float(pdf["predicted_probability"].mean())
    observed_rate = float(pdf["label"].mean())
    absolute_calibration_gap = float(abs(average_predicted - observed_rate))
    return [
        {
            "model_name": model_name,
            "run_id": run_id,
            "split_name": "test",
            "row_count": int(len(pdf)),
            "positive_rate": observed_rate,
            "mean_predicted_probability": average_predicted,
            "brier_score": brier_score,
            "absolute_calibration_gap": absolute_calibration_gap,
        }
    ]


def decile_rows(model_name: str, run_id: str, pdf: pd.DataFrame) -> list[dict[str, object]]:
    scored = append_decile_columns(pdf)
    grouped = (
        scored.groupby("score_decile", as_index=False)
        .agg(
            row_count=("label", "size"),
            observed_rate=("label", "mean"),
            mean_predicted_probability=("predicted_probability", "mean"),
            min_predicted_probability=("predicted_probability", "min"),
            max_predicted_probability=("predicted_probability", "max"),
        )
        .sort_values("score_decile")
    )
    grouped["model_name"] = model_name
    grouped["run_id"] = run_id
    grouped["split_name"] = "test"
    grouped["calibration_gap"] = grouped["mean_predicted_probability"] - grouped["observed_rate"]
    return grouped.to_dict("records")


def reliability_rows(model_name: str, run_id: str, pdf: pd.DataFrame) -> list[dict[str, object]]:
    scored = pdf.copy()
    bin_edges = [i / 10 for i in range(11)]
    scored["probability_bin"] = pd.cut(
        scored["predicted_probability"],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
        right=True,
    )
    grouped = (
        scored.groupby("probability_bin", dropna=False, as_index=False)
        .agg(
            row_count=("label", "size"),
            observed_rate=("label", "mean"),
            mean_predicted_probability=("predicted_probability", "mean"),
        )
        .dropna(subset=["probability_bin"])
        .sort_values("probability_bin")
    )
    grouped["model_name"] = model_name
    grouped["run_id"] = run_id
    grouped["split_name"] = "test"
    grouped["bin_lower_bound"] = grouped["probability_bin"].astype(int) / 10.0
    grouped["bin_upper_bound"] = grouped["bin_lower_bound"] + 0.1
    return grouped.to_dict("records")


def create_df(rows: list[dict[str, object]], schema: T.StructType):
    return spark.createDataFrame(rows, schema=schema).withColumn("processed_at_utc", F.current_timestamp())


def main() -> None:
    summary_rows: list[dict[str, object]] = []
    decile_output_rows: list[dict[str, object]] = []
    reliability_output_rows: list[dict[str, object]] = []
    plot_frames: list[pd.DataFrame] = []

    selected_scores = selected_test_scores()
    for row in selected_scores.select("model_name", "run_id").distinct().orderBy("model_name").collect():
        model_name = row["model_name"]
        run_id = row["run_id"]
        pdf = (
            selected_scores.filter(F.col("model_name") == F.lit(model_name))
            .filter(F.col("run_id") == F.lit(run_id))
            .select("label", "predicted_probability")
            .toPandas()
        )
        summary_rows.extend(calibration_summary_rows(model_name, run_id, pdf))
        decile_output_rows.extend(decile_rows(model_name, run_id, pdf))
        reliability_output_rows.extend(reliability_rows(model_name, run_id, pdf))
        plot_frame = pd.DataFrame(reliability_rows(model_name, run_id, pdf))
        plot_frames.append(plot_frame)

    summary_schema = T.StructType(
        [
            T.StructField("model_name", T.StringType(), False),
            T.StructField("run_id", T.StringType(), False),
            T.StructField("split_name", T.StringType(), False),
            T.StructField("row_count", T.LongType(), False),
            T.StructField("positive_rate", T.DoubleType(), False),
            T.StructField("mean_predicted_probability", T.DoubleType(), False),
            T.StructField("brier_score", T.DoubleType(), False),
            T.StructField("absolute_calibration_gap", T.DoubleType(), False),
            T.StructField("processed_at_utc", T.TimestampType(), True),
        ]
    )
    decile_schema = T.StructType(
        [
            T.StructField("model_name", T.StringType(), False),
            T.StructField("run_id", T.StringType(), False),
            T.StructField("split_name", T.StringType(), False),
            T.StructField("score_decile", T.LongType(), False),
            T.StructField("row_count", T.LongType(), False),
            T.StructField("observed_rate", T.DoubleType(), False),
            T.StructField("mean_predicted_probability", T.DoubleType(), False),
            T.StructField("min_predicted_probability", T.DoubleType(), False),
            T.StructField("max_predicted_probability", T.DoubleType(), False),
            T.StructField("calibration_gap", T.DoubleType(), False),
            T.StructField("processed_at_utc", T.TimestampType(), True),
        ]
    )
    reliability_schema = T.StructType(
        [
            T.StructField("model_name", T.StringType(), False),
            T.StructField("run_id", T.StringType(), False),
            T.StructField("split_name", T.StringType(), False),
            T.StructField("probability_bin", T.LongType(), False),
            T.StructField("row_count", T.LongType(), False),
            T.StructField("observed_rate", T.DoubleType(), False),
            T.StructField("mean_predicted_probability", T.DoubleType(), False),
            T.StructField("bin_lower_bound", T.DoubleType(), False),
            T.StructField("bin_upper_bound", T.DoubleType(), False),
            T.StructField("processed_at_utc", T.TimestampType(), True),
        ]
    )

    summary_df = create_df(summary_rows, summary_schema)
    decile_df = create_df(decile_output_rows, decile_schema)
    reliability_df = create_df(reliability_output_rows, reliability_schema)

    (
        summary_df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{MODEL_DATABASE}.{CALIBRATION_SUMMARY_TABLE}")
    )
    (
        decile_df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{MODEL_DATABASE}.{CALIBRATION_DECILE_TABLE}")
    )
    (
        reliability_df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{MODEL_DATABASE}.{RELIABILITY_TABLE}")
    )

    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
    for plot_df in plot_frames:
        plt.plot(
            plot_df["mean_predicted_probability"],
            plot_df["observed_rate"],
            marker="o",
            label=plot_df["model_name"].iloc[0],
        )
    plt.title("Reliability Plot (Held-Out Test Set)")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive rate")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    display(summary_df.orderBy(F.col("brier_score").asc(), F.col("absolute_calibration_gap").asc()))
    display(decile_df.orderBy("model_name", "score_decile"))
    display(reliability_df.orderBy("model_name", "probability_bin"))


# COMMAND ----------

main()
