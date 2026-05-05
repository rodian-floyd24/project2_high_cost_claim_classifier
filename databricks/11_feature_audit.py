# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Feature Audit
# MAGIC
# MAGIC Runs a formal data-quality and drift audit on the gold modeling table before additional model tuning.

# COMMAND ----------

from __future__ import annotations

import math
import os
from datetime import datetime, timezone

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from databricks.modeling_utils import apply_threshold, compute_training_only_threshold, reject_target_leakage


GOLD_DATABASE = os.environ.get("GOLD_DATABASE", "default")
MODEL_DATABASE = os.environ.get("MODEL_DATABASE", GOLD_DATABASE)
GOLD_TABLE_NAME = "gold_beneficiary_year_features"
FEATURE_QUALITY_TABLE = "gold_feature_quality_audit"
FEATURE_YEAR_DISTRIBUTION_TABLE = "gold_feature_year_distribution_audit"
FEATURE_CATEGORY_TARGET_TABLE = "gold_feature_category_target_rate_audit"
FEATURE_DRIFT_TABLE = "gold_feature_train_test_drift_audit"
TARGET_QUANTILE = float(os.environ.get("TARGET_QUANTILE", "0.9"))
MAX_CATEGORY_LEVELS = int(os.environ.get("MAX_CATEGORY_LEVELS", "50"))
VALIDATION_BUCKET_CUTOFF = int(os.environ.get("VALIDATION_BUCKET_CUTOFF", "15"))

KEY_COLUMNS = {"bene_id", "year", "target_year"}
TARGET_COLUMNS = {"target_annual_claim_cost", "target_year_high_cost_threshold", "label"}

FEATURE_QUALITY_SCHEMA = T.StructType(
    [
        T.StructField("feature_name", T.StringType(), False),
        T.StructField("feature_type", T.StringType(), False),
        T.StructField("row_count", T.LongType(), False),
        T.StructField("null_count", T.LongType(), False),
        T.StructField("null_rate", T.DoubleType(), False),
        T.StructField("unique_count", T.LongType(), False),
        T.StructField("zero_variance_flag", T.BooleanType(), False),
        T.StructField("zero_count", T.LongType(), True),
        T.StructField("zero_rate", T.DoubleType(), True),
        T.StructField("negative_count", T.LongType(), True),
        T.StructField("impossible_value_count", T.LongType(), False),
        T.StructField("min_value", T.DoubleType(), True),
        T.StructField("max_value", T.DoubleType(), True),
        T.StructField("mean_value", T.DoubleType(), True),
        T.StructField("stddev_value", T.DoubleType(), True),
        T.StructField("audit_severity", T.StringType(), False),
        T.StructField("processed_at_utc", T.TimestampType(), False),
    ]
)

YEAR_DISTRIBUTION_SCHEMA = T.StructType(
    [
        T.StructField("year", T.IntegerType(), False),
        T.StructField("feature_name", T.StringType(), False),
        T.StructField("feature_type", T.StringType(), False),
        T.StructField("row_count", T.LongType(), False),
        T.StructField("mean_value", T.DoubleType(), True),
        T.StructField("stddev_value", T.DoubleType(), True),
        T.StructField("min_value", T.DoubleType(), True),
        T.StructField("median_value", T.DoubleType(), True),
        T.StructField("max_value", T.DoubleType(), True),
        T.StructField("category_value", T.StringType(), True),
        T.StructField("category_count", T.LongType(), True),
        T.StructField("category_rate", T.DoubleType(), True),
        T.StructField("processed_at_utc", T.TimestampType(), False),
    ]
)

CATEGORY_TARGET_SCHEMA = T.StructType(
    [
        T.StructField("feature_name", T.StringType(), False),
        T.StructField("category_value", T.StringType(), True),
        T.StructField("row_count", T.LongType(), False),
        T.StructField("positive_count", T.LongType(), False),
        T.StructField("target_rate", T.DoubleType(), True),
        T.StructField("processed_at_utc", T.TimestampType(), False),
    ]
)

DRIFT_SCHEMA = T.StructType(
    [
        T.StructField("feature_name", T.StringType(), False),
        T.StructField("feature_type", T.StringType(), False),
        T.StructField("category_value", T.StringType(), True),
        T.StructField("train_row_count", T.LongType(), False),
        T.StructField("test_row_count", T.LongType(), False),
        T.StructField("train_value", T.DoubleType(), True),
        T.StructField("test_value", T.DoubleType(), True),
        T.StructField("absolute_difference", T.DoubleType(), True),
        T.StructField("standardized_difference", T.DoubleType(), True),
        T.StructField("drift_flag", T.BooleanType(), False),
        T.StructField("processed_at_utc", T.TimestampType(), False),
    ]
)


def utc_now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


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
            F.col("current.*"),
            F.col("next.year").alias("target_year"),
            F.col("next.annual_claim_cost").alias("target_annual_claim_cost"),
        )
    )


def split_modeling_frame(df: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    target_years = [row["target_year"] for row in df.select("target_year").distinct().orderBy("target_year").collect()]
    if len(target_years) < 2:
        raise ValueError("Feature audit requires at least two prospective target years.")

    test_target_year = target_years[-1]
    training_pool = df.filter(F.col("target_year") < F.lit(test_target_year))
    test_df = df.filter(F.col("target_year") == F.lit(test_target_year)).withColumn("audit_split", F.lit("test"))

    split_assignments = training_pool.select("bene_id").distinct().withColumn(
        "shared_split_bucket",
        F.pmod(F.xxhash64("bene_id"), F.lit(100)),
    )
    train_ids = split_assignments.filter(F.col("shared_split_bucket") >= F.lit(VALIDATION_BUCKET_CUTOFF)).select(
        "bene_id"
    )
    validation_ids = split_assignments.filter(F.col("shared_split_bucket") < F.lit(VALIDATION_BUCKET_CUTOFF)).select(
        "bene_id"
    )
    train_df = training_pool.join(train_ids, "bene_id", "inner").withColumn("audit_split", F.lit("train"))
    validation_df = training_pool.join(validation_ids, "bene_id", "inner").withColumn("audit_split", F.lit("validation"))
    return train_df, validation_df, test_df


def add_training_target(train_df: DataFrame, validation_df: DataFrame, test_df: DataFrame) -> DataFrame:
    threshold = compute_training_only_threshold(train_df, TARGET_QUANTILE)
    return (
        apply_threshold(train_df, threshold)
        .unionByName(apply_threshold(validation_df, threshold))
        .unionByName(apply_threshold(test_df, threshold))
    )


def feature_columns(df: DataFrame) -> list[str]:
    excluded = KEY_COLUMNS | TARGET_COLUMNS | {"audit_split"}
    return [column_name for column_name in df.columns if column_name not in excluded]


def is_numeric_type(data_type: T.DataType) -> bool:
    return isinstance(
        data_type,
        (
            T.ByteType,
            T.ShortType,
            T.IntegerType,
            T.LongType,
            T.FloatType,
            T.DoubleType,
            T.DecimalType,
        ),
    )


def numeric_columns(df: DataFrame, columns: list[str]) -> list[str]:
    type_by_name = {field.name: field.dataType for field in df.schema.fields}
    return [column_name for column_name in columns if is_numeric_type(type_by_name[column_name])]


def categorical_columns(df: DataFrame, columns: list[str]) -> list[str]:
    return [column_name for column_name in columns if column_name not in set(numeric_columns(df, columns))]


def impossible_condition(column_name: str):
    column = F.col(column_name)
    lower_name = column_name.lower()

    if lower_name in {"sex", "age_band", "age_5yr_band", "race_code", "state_code"}:
        return F.lit(False)
    if lower_name.endswith("_flag") or lower_name.endswith("_indicator") or lower_name.startswith("any_"):
        return column.isNotNull() & (~column.isin(0, 1))
    if "share" in lower_name or lower_name.endswith("_fraction") or "percentile" in lower_name:
        return column.isNotNull() & ((column < 0) | (column > 1))
    if lower_name.endswith("month") or lower_name.endswith("months") or "months_count" in lower_name:
        return column.isNotNull() & ((column < 0) | (column > 12))
    if lower_name in {"age_years", "age_years_imputed"}:
        return column.isNotNull() & ((column < 0) | (column > 115))
    if (
        "count" in lower_name
        or "cost" in lower_name
        or "amount" in lower_name
        or "total" in lower_name
        or "days_supply" in lower_name
    ) and "change" not in lower_name and "difference" not in lower_name:
        return column.isNotNull() & (column < 0)
    return F.lit(False)


def feature_quality_rows(df: DataFrame, columns: list[str]) -> DataFrame:
    total_rows = df.count()
    rows = []
    numeric_set = set(numeric_columns(df, columns))
    processed_at = utc_now()

    for column_name in columns:
        base_aggs = [
            F.sum(F.when(F.col(column_name).isNull(), 1).otherwise(0)).alias("null_count"),
            F.countDistinct(F.col(column_name)).alias("unique_count"),
            F.sum(F.when(impossible_condition(column_name), 1).otherwise(0)).alias("impossible_value_count"),
        ]
        if column_name in numeric_set:
            base_aggs.extend(
                [
                    F.min(F.col(column_name)).alias("min_value"),
                    F.max(F.col(column_name)).alias("max_value"),
                    F.avg(F.col(column_name).cast("double")).alias("mean_value"),
                    F.stddev(F.col(column_name).cast("double")).alias("stddev_value"),
                    F.sum(F.when(F.col(column_name) == 0, 1).otherwise(0)).alias("zero_count"),
                    F.sum(F.when(F.col(column_name) < 0, 1).otherwise(0)).alias("negative_count"),
                ]
            )
        else:
            base_aggs.extend(
                [
                    F.lit(None).cast("double").alias("min_value"),
                    F.lit(None).cast("double").alias("max_value"),
                    F.lit(None).cast("double").alias("mean_value"),
                    F.lit(None).cast("double").alias("stddev_value"),
                    F.lit(None).cast("long").alias("zero_count"),
                    F.lit(None).cast("long").alias("negative_count"),
                ]
            )

        row = df.agg(*base_aggs).collect()[0].asDict()
        unique_count = int(row["unique_count"] or 0)
        null_count = int(row["null_count"] or 0)
        impossible_count = int(row["impossible_value_count"] or 0)
        zero_variance = unique_count <= 1
        severity = "pass"
        if impossible_count > 0:
            severity = "fail"
        elif zero_variance:
            severity = "warn"
        elif null_count > 0:
            severity = "warn"

        rows.append(
            {
                "feature_name": column_name,
                "feature_type": "numeric" if column_name in numeric_set else "categorical",
                "row_count": total_rows,
                "null_count": null_count,
                "null_rate": 0.0 if total_rows == 0 else null_count / total_rows,
                "unique_count": unique_count,
                "zero_variance_flag": zero_variance,
                "zero_count": row["zero_count"],
                "zero_rate": None
                if row["zero_count"] is None or total_rows == 0
                else float(row["zero_count"]) / total_rows,
                "negative_count": row["negative_count"],
                "impossible_value_count": impossible_count,
                "min_value": row["min_value"],
                "max_value": row["max_value"],
                "mean_value": row["mean_value"],
                "stddev_value": row["stddev_value"],
                "audit_severity": severity,
                "processed_at_utc": processed_at,
            }
        )

    return spark.createDataFrame(rows, schema=FEATURE_QUALITY_SCHEMA)


def numeric_year_distribution_rows(df: DataFrame, columns: list[str]) -> DataFrame:
    rows = []
    processed_at = utc_now()
    for column_name in columns:
        for row in (
            df.groupBy("year")
            .agg(
                F.count("*").alias("row_count"),
                F.avg(F.col(column_name).cast("double")).alias("mean_value"),
                F.stddev(F.col(column_name).cast("double")).alias("stddev_value"),
                F.min(F.col(column_name)).alias("min_value"),
                F.expr(f"percentile_approx({column_name}, 0.5)").alias("median_value"),
                F.max(F.col(column_name)).alias("max_value"),
            )
            .collect()
        ):
            row_dict = row.asDict()
            row_dict.update(
                {
                    "feature_name": column_name,
                    "feature_type": "numeric",
                    "category_value": None,
                    "category_count": None,
                    "category_rate": None,
                    "processed_at_utc": processed_at,
                }
            )
            rows.append(row_dict)
    return spark.createDataFrame(rows, schema=YEAR_DISTRIBUTION_SCHEMA)


def categorical_year_distribution_rows(df: DataFrame, columns: list[str]) -> DataFrame:
    rows = []
    processed_at = utc_now()
    year_counts = {row["year"]: row["row_count"] for row in df.groupBy("year").agg(F.count("*").alias("row_count")).collect()}
    for column_name in columns:
        distinct_count = df.select(column_name).distinct().count()
        if distinct_count > MAX_CATEGORY_LEVELS:
            continue
        for row in df.groupBy("year", column_name).agg(F.count("*").alias("category_count")).collect():
            row_dict = row.asDict()
            row_count = int(year_counts[row_dict["year"]])
            rows.append(
                {
                    "year": row_dict["year"],
                    "feature_name": column_name,
                    "feature_type": "categorical",
                    "row_count": row_count,
                    "mean_value": None,
                    "stddev_value": None,
                    "min_value": None,
                    "median_value": None,
                    "max_value": None,
                    "category_value": str(row_dict[column_name]),
                    "category_count": int(row_dict["category_count"]),
                    "category_rate": 0.0 if row_count == 0 else int(row_dict["category_count"]) / row_count,
                    "processed_at_utc": processed_at,
                }
            )
    return spark.createDataFrame(rows, schema=YEAR_DISTRIBUTION_SCHEMA)


def category_target_rate_rows(df: DataFrame, columns: list[str]) -> DataFrame:
    rows = []
    processed_at = utc_now()
    for column_name in columns:
        distinct_count = df.select(column_name).distinct().count()
        if distinct_count > MAX_CATEGORY_LEVELS:
            continue
        for row in (
            df.groupBy(column_name)
            .agg(
                F.count("*").alias("row_count"),
                F.avg("label").alias("target_rate"),
                F.sum("label").cast("long").alias("positive_count"),
            )
            .collect()
        ):
            row_dict = row.asDict()
            rows.append(
                {
                    "feature_name": column_name,
                    "category_value": str(row_dict[column_name]),
                    "row_count": int(row_dict["row_count"]),
                    "positive_count": int(row_dict["positive_count"] or 0),
                    "target_rate": row_dict["target_rate"],
                    "processed_at_utc": processed_at,
                }
            )
    return spark.createDataFrame(rows, schema=CATEGORY_TARGET_SCHEMA)


def drift_rows(df: DataFrame, numeric_feature_columns: list[str], categorical_feature_columns: list[str]) -> DataFrame:
    rows = []
    processed_at = utc_now()
    split_counts = {row["audit_split"]: row["row_count"] for row in df.groupBy("audit_split").agg(F.count("*").alias("row_count")).collect()}
    train_count = split_counts.get("train", 0)
    test_count = split_counts.get("test", 0)

    for column_name in numeric_feature_columns:
        stats = {
            row["audit_split"]: row.asDict()
            for row in df.groupBy("audit_split")
            .agg(
                F.avg(F.col(column_name).cast("double")).alias("mean_value"),
                F.stddev(F.col(column_name).cast("double")).alias("stddev_value"),
            )
            .collect()
        }
        train = stats.get("train", {})
        test = stats.get("test", {})
        train_mean = train.get("mean_value")
        test_mean = test.get("mean_value")
        train_std = train.get("stddev_value") or 0.0
        test_std = test.get("stddev_value") or 0.0
        pooled_std = math.sqrt((train_std**2 + test_std**2) / 2.0)
        standardized_difference = None if pooled_std == 0 or train_mean is None or test_mean is None else (test_mean - train_mean) / pooled_std
        rows.append(
            {
                "feature_name": column_name,
                "feature_type": "numeric",
                "category_value": None,
                "train_row_count": train_count,
                "test_row_count": test_count,
                "train_value": train_mean,
                "test_value": test_mean,
                "absolute_difference": None if train_mean is None or test_mean is None else abs(test_mean - train_mean),
                "standardized_difference": standardized_difference,
                "drift_flag": bool(standardized_difference is not None and abs(standardized_difference) >= 0.1),
                "processed_at_utc": processed_at,
            }
        )

    for column_name in categorical_feature_columns:
        distinct_count = df.select(column_name).distinct().count()
        if distinct_count > MAX_CATEGORY_LEVELS:
            continue
        counts = (
            df.groupBy("audit_split", column_name)
            .agg(F.count("*").alias("row_count"))
            .withColumn(
                "rate",
                F.when(
                    F.col("audit_split") == "train",
                    F.col("row_count") / F.lit(train_count if train_count else 1),
                ).otherwise(F.col("row_count") / F.lit(test_count if test_count else 1)),
            )
        )
        train_rates = {
            str(row[column_name]): row["rate"]
            for row in counts.filter(F.col("audit_split") == "train").collect()
        }
        test_rates = {
            str(row[column_name]): row["rate"]
            for row in counts.filter(F.col("audit_split") == "test").collect()
        }
        for category_value in sorted(set(train_rates) | set(test_rates)):
            train_rate = float(train_rates.get(category_value, 0.0))
            test_rate = float(test_rates.get(category_value, 0.0))
            rows.append(
                {
                    "feature_name": column_name,
                    "feature_type": "categorical",
                    "category_value": category_value,
                    "train_row_count": train_count,
                    "test_row_count": test_count,
                    "train_value": train_rate,
                    "test_value": test_rate,
                    "absolute_difference": abs(test_rate - train_rate),
                    "standardized_difference": None,
                    "drift_flag": abs(test_rate - train_rate) >= 0.05,
                    "processed_at_utc": processed_at,
                }
            )
    return spark.createDataFrame(rows, schema=DRIFT_SCHEMA)


def write_table(df: DataFrame, table_name: str) -> None:
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{MODEL_DATABASE}.{table_name}")
    )


def main() -> None:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {MODEL_DATABASE}")
    train_df, validation_df, test_df = split_modeling_frame(build_modeling_frame(read_gold()))
    audited_df = add_training_target(train_df, validation_df, test_df).cache()
    columns = feature_columns(audited_df)
    reject_target_leakage(columns)
    numeric_feature_columns = numeric_columns(audited_df, columns)
    categorical_feature_columns = categorical_columns(audited_df, columns)

    quality_df = feature_quality_rows(audited_df, columns)
    year_distribution_df = numeric_year_distribution_rows(audited_df, numeric_feature_columns).unionByName(
        categorical_year_distribution_rows(audited_df, categorical_feature_columns),
        allowMissingColumns=True,
    )
    category_target_df = category_target_rate_rows(audited_df, categorical_feature_columns)
    drift_df = drift_rows(audited_df, numeric_feature_columns, categorical_feature_columns)

    write_table(quality_df, FEATURE_QUALITY_TABLE)
    write_table(year_distribution_df, FEATURE_YEAR_DISTRIBUTION_TABLE)
    write_table(category_target_df, FEATURE_CATEGORY_TARGET_TABLE)
    write_table(drift_df, FEATURE_DRIFT_TABLE)

    print(f"feature quality audit written to {MODEL_DATABASE}.{FEATURE_QUALITY_TABLE}")
    print(f"year distribution audit written to {MODEL_DATABASE}.{FEATURE_YEAR_DISTRIBUTION_TABLE}")
    print(f"category target-rate audit written to {MODEL_DATABASE}.{FEATURE_CATEGORY_TARGET_TABLE}")
    print(f"train/test drift audit written to {MODEL_DATABASE}.{FEATURE_DRIFT_TABLE}")

    display(quality_df.orderBy(F.col("audit_severity").desc(), F.col("feature_name")))
    display(drift_df.filter(F.col("drift_flag")).orderBy(F.col("absolute_difference").desc()))


# COMMAND ----------

main()
