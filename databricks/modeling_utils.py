from __future__ import annotations

TARGET_COLUMNS = {
    "target_annual_claim_cost",
    "target_year_high_cost_threshold",
    "target_high_cost_threshold",
    "target_cost_within_year_percentile",
    "label",
}

FUTURE_FEATURE_PREFIXES = ("target_", "next_year_")
SHARED_SPLIT_VERSION = "xxhash64_bene_id_mod_100_v2_beneficiary_hash_holdout"
TEST_BUCKET_CUTOFF = 15
VALIDATION_BUCKET_CUTOFF = 30
TARGET_QUANTILE = 0.9


def validate_required_columns(df, required_columns: list[str]) -> list[str]:
    """Return missing columns; callers decide whether to warn or fail."""

    return sorted(set(required_columns) - set(df.columns))


def validate_unique_key(df, key_columns: list[str]) -> int:
    from pyspark.sql import functions as F

    nonnull_count = df.filter(
        " AND ".join(f"{column} IS NOT NULL" for column in key_columns)
    ).count()
    distinct_count = df.select(*key_columns).dropna().distinct().count()
    return int(nonnull_count - distinct_count)


def validate_no_missing_key(df, key_columns: list[str]) -> int:
    from functools import reduce
    from operator import or_

    from pyspark.sql import functions as F

    missing_expr = reduce(or_, [F.col(column).isNull() for column in key_columns])
    return int(df.filter(missing_expr).count())


def build_year_t_to_t_plus_1_frame(gold_df):
    from pyspark.sql import functions as F

    current_year = gold_df.alias("current")
    next_year = gold_df.select("bene_id", "year", "annual_claim_cost").alias("next")
    return (
        current_year.join(
            next_year,
            (F.col("current.bene_id") == F.col("next.bene_id"))
            & (F.col("current.year") + F.lit(1) == F.col("next.year")),
            "inner",
        )
        .select(
            F.col("current.*"),
            F.col("current.year").alias("feature_year"),
            F.col("next.year").alias("target_year"),
            F.col("next.annual_claim_cost").alias("target_annual_claim_cost"),
        )
    )


def assign_beneficiary_hash_holdout_split(modeling_df):
    from pyspark.sql import functions as F

    split_assignments = modeling_df.select("bene_id").distinct().withColumn(
        "shared_split_bucket",
        F.pmod(F.xxhash64("bene_id"), F.lit(100)),
    )
    train_ids = split_assignments.filter(F.col("shared_split_bucket") >= F.lit(VALIDATION_BUCKET_CUTOFF)).select(
        "bene_id"
    )
    validation_ids = split_assignments.filter(
        (F.col("shared_split_bucket") >= F.lit(TEST_BUCKET_CUTOFF))
        & (F.col("shared_split_bucket") < F.lit(VALIDATION_BUCKET_CUTOFF))
    ).select("bene_id")
    test_ids = split_assignments.filter(F.col("shared_split_bucket") < F.lit(TEST_BUCKET_CUTOFF)).select(
        "bene_id"
    )
    train_df = modeling_df.join(train_ids, "bene_id", "inner").withColumn("split_name", F.lit("train"))
    validation_df = modeling_df.join(validation_ids, "bene_id", "inner").withColumn(
        "split_name", F.lit("validation")
    )
    test_df = modeling_df.join(test_ids, "bene_id", "inner").withColumn("split_name", F.lit("test"))
    return train_df.unionByName(validation_df).unionByName(test_df)


def split_train_validation_test(modeling_df):
    split_df = assign_beneficiary_hash_holdout_split(modeling_df)
    return (
        split_df.filter("split_name = 'train'"),
        split_df.filter("split_name = 'validation'"),
        split_df.filter("split_name = 'test'"),
    )


def compute_top_decile_threshold(train_df) -> float:
    from pyspark.sql import functions as F

    row = train_df.agg(
        F.expr(f"percentile_approx(target_annual_claim_cost, {TARGET_QUANTILE})").alias("threshold")
    ).collect()[0]
    return float(row["threshold"])


def apply_binary_label(modeling_df, threshold: float):
    from pyspark.sql import functions as F

    return (
        modeling_df.withColumn("target_high_cost_threshold", F.lit(float(threshold)))
        .withColumn("target_year_high_cost_threshold", F.lit(float(threshold)))
        .withColumn(
            "label",
            F.when(F.col("target_annual_claim_cost") >= F.lit(float(threshold)), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
    )


def reject_target_leakage(feature_columns: list[str]) -> None:
    leaking_columns = [
        column
        for column in feature_columns
        if column in TARGET_COLUMNS or any(column.startswith(prefix) for prefix in FUTURE_FEATURE_PREFIXES)
    ]
    if leaking_columns:
        raise ValueError(f"Target leakage columns are not allowed as features: {sorted(leaking_columns)}")


def build_prospective_modeling_frame(gold_df):
    return build_year_t_to_t_plus_1_frame(gold_df)


def assign_shared_split(modeling_df):
    return assign_beneficiary_hash_holdout_split(modeling_df)


def compute_training_only_threshold(train_df, quantile: float = TARGET_QUANTILE) -> float:
    from pyspark.sql import functions as F

    row = train_df.agg(
        F.expr(f"percentile_approx(target_annual_claim_cost, {float(quantile)})").alias("threshold")
    ).collect()[0]
    return float(row["threshold"])


def apply_threshold(modeling_df, threshold: float):
    return apply_binary_label(modeling_df, threshold)


def validate_no_target_leakage(feature_columns: list[str]) -> None:
    reject_target_leakage(feature_columns)
