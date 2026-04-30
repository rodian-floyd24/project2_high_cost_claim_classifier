# Databricks notebook source
# MAGIC %md
# MAGIC # Train Paper-Aligned XGBoost Baseline
# MAGIC
# MAGIC Trains an XGBoost baseline following the Chen and Guestrin regularized boosted-tree framing: logistic objective, shrinkage, structural regularization, column subsampling, and validation early stopping. Ranking and top-k targeting metrics are reported as downstream evaluation outputs.

# COMMAND ----------

# MAGIC %pip install xgboost

# COMMAND ----------

from __future__ import annotations

import math
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from pyspark.sql import DataFrame
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql import types as T
from xgboost import XGBClassifier


GOLD_DATABASE = os.environ.get("GOLD_DATABASE", "default")
MODEL_DATABASE = os.environ.get("MODEL_DATABASE", GOLD_DATABASE)
GOLD_TABLE_NAME = "gold_beneficiary_year_features"
TRAINING_AUDIT_TABLE = "xgboost_training_audit"
TOPK_CURVE_TABLE = "model_topk_curve_points"
PREDICTION_SCORE_TABLE = "model_prediction_scores"
MODEL_NAME = "xgboost"
RANDOM_SEED = 42
DEFAULT_MLFLOW_EXPERIMENT_PATH = "/Shared/Project2HighCostClaimClassifier_Experiment"
MLFLOW_EXPERIMENT_PATH = os.environ.get("MLFLOW_EXPERIMENT_PATH", DEFAULT_MLFLOW_EXPERIMENT_PATH)
MAX_PANDAS_ROWS_PER_SPLIT = int(os.environ.get("MAX_PANDAS_ROWS_PER_SPLIT", "500000"))
EARLY_STOPPING_ROUNDS = 50
FINAL_N_ESTIMATORS_UPPER_BOUND = 1000
FIXED_DECISION_THRESHOLD = 0.5
TARGET_QUANTILE = 0.9
SPLIT_STRATEGY = "temporal_target_year_holdout"
SHARED_SPLIT_VERSION = "xxhash64_bene_id_mod_100_v1"
VALIDATION_BUCKET_CUTOFF = 15
CALIBRATION_METHOD = os.environ.get("XGBOOST_CALIBRATION_METHOD", "isotonic")

PAPER_ALIGNED_XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    # The paper presents exact greedy and approximate split finding. Histogram
    # mode is used here as the scalable approximate/binning implementation choice.
    "tree_method": "hist",
    "n_estimators": FINAL_N_ESTIMATORS_UPPER_BOUND,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
}

CHRONIC_FLAG_FEATURES = [
    "alzheimers_flag",
    "chf_flag",
    "chronic_kidney_disease_flag",
    "cancer_flag",
    "copd_flag",
    "depression_flag",
    "diabetes_flag",
    "ischemic_heart_disease_flag",
    "osteoporosis_flag",
    "rheumatoid_arthritis_oa_flag",
    "stroke_tia_flag",
]

NUMERIC_FEATURES = [
    "enrollment_months_count",
    "full_year_enrolled_flag",
    "partial_year_enrolled_flag",
    "zero_enrollment_flag",
    "low_enrollment_flag",
    "enrollment_fraction",
    "annualized_cost_per_enrolled_month",
    "annualized_claims_per_enrolled_month",
    "age_years",
    "age_missing_flag",
    "age_years_imputed",
    "age_over_65",
    "age_over_75",
    "age_over_85",
    "age_squared",
    "chronic_condition_count_squared",
    "chronic_condition_count",
    *CHRONIC_FLAG_FEATURES,
    "claims_per_month_chronic_count_interaction",
    "providers_per_month_chronic_count_interaction",
    "age_inpatient_claim_interaction",
    "age_total_claim_interaction",
    "age_chronic_count_interaction",
    "sex_male_chronic_count_interaction",
    "sex_female_chronic_count_interaction",
    "enrollment_months_total_claim_interaction",
    "enrollment_months_inpatient_interaction",
    "chronic_count_age_under_65",
    "chronic_count_age_65_74",
    "chronic_count_age_75_84",
    "chronic_count_age_85_plus",
    "inpatient_claim_count",
    "outpatient_claim_count",
    "carrier_claim_count",
    "pde_claim_count",
    "outpatient_ed_claim_count",
    "outpatient_line_count",
    "carrier_line_count",
    "rx_days_supply",
    "total_claim_days",
    "total_claim_count",
    "unique_provider_count",
    "cost_per_enrollment_month",
    "claims_per_enrollment_month",
    "claim_days_per_enrollment_month",
    "providers_per_enrollment_month",
    "provider_fragmentation_index",
    "inpatient_claims_per_enrollment_month",
    "outpatient_claims_per_enrollment_month",
    "carrier_claims_per_enrollment_month",
    "rx_fills_per_enrollment_month",
    "outpatient_ed_claims_per_enrollment_month",
    "rx_days_supply_per_enrollment_month",
    "avg_inpatient_cost_per_claim",
    "avg_outpatient_cost_per_claim",
    "avg_carrier_cost_per_claim",
    "avg_rx_cost_per_fill",
    "outpatient_lines_per_claim",
    "carrier_lines_per_claim",
    "any_inpatient_claim",
    "any_outpatient_claim",
    "any_carrier_claim",
    "any_pde_claim",
    "any_outpatient_ed_claim",
    "multiple_provider_flag",
    "multi_setting_utilization_flag",
    "rx_total_cost",
    "inpatient_total_cost",
    "outpatient_total_cost",
    "carrier_total_cost",
    "carrier_allowed_total",
    "rx_patient_pay_total",
    "rx_cost_log1p",
    "inpatient_cost_log1p",
    "outpatient_cost_log1p",
    "carrier_cost_log1p",
    "annual_cost_log1p",
    "inpatient_claim_count_log1p",
    "outpatient_claim_count_log1p",
    "carrier_claim_count_log1p",
    "pde_claim_count_log1p",
    "total_claim_count_log1p",
    "unique_provider_count_log1p",
    "annual_cost_year_percentile",
    "annual_cost_year_decile",
    "annual_cost_to_year_median",
    "has_prior_year",
    "prior_year_annual_claim_cost",
    "prior_year_inpatient_claim_count",
    "prior_year_total_claim_count",
    "prior_year_enrollment_months_count",
    "current_year_high_cost_indicator",
    "prior_year_high_cost_indicator",
    "two_year_avg_annual_claim_cost",
    "cost_trend_difference",
    "cost_trend_ratio",
    "inpatient_claim_count_change",
    "total_claim_count_change",
    "high_cost_last_2yr_count",
    "high_cost_1_of_last_2yr",
    "high_cost_2_of_last_2yr",
    "inpatient_cost_share",
    "outpatient_cost_share",
    "carrier_cost_share",
    "rx_cost_share",
]

CATEGORICAL_FEATURES = [
    "age_band",
    "sex",
    "race_code",
    "state_code",
    "chronic_burden_band",
    "chronic_burden_age_band",
    "age_5yr_band",
    "chronic_burden_age_5yr_band",
    "sex_chronic_burden_band",
    "enrollment_months_band",
    "utilization_trend",
]
REQUIRED_GOLD_COLUMNS = ["bene_id", "year", "annual_claim_cost", *(NUMERIC_FEATURES + CATEGORICAL_FEATURES)]
REQUIRED_MODELING_COLUMNS = [*REQUIRED_GOLD_COLUMNS, "target_year", "target_annual_claim_cost"]


def read_gold() -> DataFrame:
    df = spark.table(f"{GOLD_DATABASE}.{GOLD_TABLE_NAME}")
    validate_required_columns(df, REQUIRED_GOLD_COLUMNS, f"{GOLD_DATABASE}.{GOLD_TABLE_NAME}")
    validate_unique_keys(df, ["bene_id", "year"], f"{GOLD_DATABASE}.{GOLD_TABLE_NAME}")
    return df


def validate_required_columns(df: DataFrame, required_columns: list[str], df_name: str) -> None:
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"{df_name} is missing required columns: {missing_columns}")


def validate_unique_keys(df: DataFrame, key_columns: list[str], df_name: str) -> None:
    duplicate_count = df.groupBy(*key_columns).count().filter(F.col("count") > F.lit(1)).limit(1).count()
    if duplicate_count:
        raise ValueError(f"{df_name} contains duplicate rows for key columns: {key_columns}")


def require_nonempty_split(df: DataFrame, split_name: str) -> None:
    if not df.take(1):
        raise ValueError(f"{split_name} split is empty.")


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


def split_gold_by_time(df: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    target_years = [row["target_year"] for row in df.select("target_year").distinct().orderBy("target_year").collect()]
    if len(target_years) < 2:
        raise ValueError("Temporal holdout requires at least two target years.")

    test_target_year = target_years[-1]
    training_pool = df.filter(F.col("target_year") < F.lit(test_target_year))
    test_df = df.filter(F.col("target_year") == F.lit(test_target_year))
    split_assignments = training_pool.select("bene_id").distinct().withColumn(
        "shared_split_bucket",
        F.pmod(F.xxhash64("bene_id"), F.lit(100)),
    )
    train_ids = split_assignments.filter(F.col("shared_split_bucket") >= F.lit(VALIDATION_BUCKET_CUTOFF)).select("bene_id")
    validation_ids = split_assignments.filter(F.col("shared_split_bucket") < F.lit(VALIDATION_BUCKET_CUTOFF)).select(
        "bene_id"
    )
    train_df = training_pool.join(train_ids, "bene_id", "inner")
    validation_df = training_pool.join(validation_ids, "bene_id", "inner")
    require_nonempty_split(train_df, "train")
    require_nonempty_split(validation_df, "validation")
    require_nonempty_split(test_df, "test")
    return train_df, validation_df, test_df


def add_training_target(
    train_df: DataFrame,
    validation_df: DataFrame,
    test_df: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame, float]:
    combined_df = train_df.unionByName(validation_df).unionByName(test_df)
    threshold_df = combined_df.groupBy("target_year").agg(
        F.expr(f"percentile_approx(target_annual_claim_cost, {TARGET_QUANTILE})").alias(
            "target_year_high_cost_threshold"
        )
    )
    threshold_summary = threshold_df.agg(F.avg("target_year_high_cost_threshold").alias("threshold")).collect()[0]
    threshold = float(threshold_summary["threshold"])

    def with_target(df: DataFrame) -> DataFrame:
        return (
            df.join(threshold_df, "target_year", "left")
            .withColumn(
                "target_cost_within_year_percentile",
                F.percent_rank().over(Window.partitionBy("target_year").orderBy(F.col("target_annual_claim_cost"))),
            )
            .withColumn(
                "label",
                F.when(
                    F.col("target_annual_claim_cost") > F.col("target_year_high_cost_threshold"),
                    F.lit(1.0),
                ).otherwise(F.lit(0.0)),
            )
        )

    return with_target(train_df), with_target(validation_df), with_target(test_df), float(threshold)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def compute_scale_pos_weight(y_true: pd.Series) -> float:
    positive_count = int(y_true.sum())
    negative_count = int(len(y_true) - positive_count)
    if positive_count <= 0:
        return 1.0
    return float(negative_count / positive_count)


def fit_final_pipeline_with_early_stopping(
    scale_pos_weight: float,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> Pipeline:
    preprocessor = clone(build_preprocessor())
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_validation_transformed = preprocessor.transform(x_validation)

    final_classifier = XGBClassifier(
        **PAPER_ALIGNED_XGBOOST_PARAMS,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    final_classifier.fit(
        x_train_transformed,
        y_train,
        eval_set=[(x_validation_transformed, y_validation)],
        verbose=False,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", final_classifier),
        ]
    )


def calibrate_prefit_model(model: Pipeline, x_calibration: pd.DataFrame, y_calibration: pd.Series) -> CalibratedClassifierCV:
    if CALIBRATION_METHOD not in {"sigmoid", "isotonic"}:
        raise ValueError("XGBOOST_CALIBRATION_METHOD must be either 'sigmoid' for Platt scaling or 'isotonic'.")
    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method=CALIBRATION_METHOD,
        cv="prefit",
    )
    calibrated_model.fit(x_calibration, y_calibration)
    return calibrated_model


def to_pandas_features(df: DataFrame) -> pd.DataFrame:
    row_count = df.count()
    if row_count > MAX_PANDAS_ROWS_PER_SPLIT:
        raise ValueError(
            f"Refusing to collect {row_count} rows to pandas. "
            f"Set MAX_PANDAS_ROWS_PER_SPLIT above {row_count} if local training is intentional."
        )
    if row_count == 0:
        raise ValueError("Refusing to collect an empty split to pandas.")

    return df.select(
        "bene_id",
        "year",
        "target_year",
        "target_annual_claim_cost",
        *(NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["label"]),
    ).toPandas()


def validate_label_distribution(y_true: pd.Series, split_name: str, require_both_classes: bool) -> None:
    class_count = int(y_true.nunique())
    if require_both_classes and class_count < 2:
        raise ValueError(f"{split_name} split must contain both positive and negative labels.")


def binary_metric_or_none(metric_fn, y_true, y_score) -> float | None:
    if pd.Series(y_true).nunique() < 2:
        return None
    return float(metric_fn(y_true, y_score))


def top_k_capture_and_lift(y_true, y_score, top_fraction: float) -> tuple[float, float]:
    top_n = max(1, int(math.ceil(len(y_true) * top_fraction)))
    ranked = pd.DataFrame({"label": y_true, "score": y_score}).sort_values("score", ascending=False)
    selected = ranked.head(top_n)
    total_positives = float(ranked["label"].sum())
    base_rate = float(ranked["label"].mean())
    selected_rate = float(selected["label"].mean())
    capture = 0.0 if total_positives == 0 else float(selected["label"].sum()) / total_positives
    lift = 0.0 if base_rate == 0 else selected_rate / base_rate
    return capture, lift


def evaluate_predictions(y_true, y_score, y_pred, split_name: str) -> dict[str, float | int | str | None]:
    capture_5, lift_5 = top_k_capture_and_lift(y_true, y_score, 0.05)
    capture_10, lift_10 = top_k_capture_and_lift(y_true, y_score, 0.10)
    return {
        "split_name": split_name,
        "row_count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "area_under_roc": binary_metric_or_none(roc_auc_score, y_true, y_score),
        "area_under_pr": binary_metric_or_none(average_precision_score, y_true, y_score),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "positive_rate": float(y_true.mean()),
        "top_5_capture_rate": float(capture_5),
        "top_5_lift": float(lift_5),
        "top_10_capture_rate": float(capture_10),
        "top_10_lift": float(lift_10),
    }


def create_audit_df(
    metrics: list[dict[str, float | int | str | None]],
    label_threshold: float,
    decision_threshold: float,
    run_id: str,
) -> DataFrame:
    schema = T.StructType(
        [
            T.StructField("run_id", T.StringType(), False),
            T.StructField("split_name", T.StringType(), False),
            T.StructField("row_count", T.LongType(), False),
            T.StructField("positive_rate", T.DoubleType(), False),
            T.StructField("accuracy", T.DoubleType(), False),
            T.StructField("precision", T.DoubleType(), False),
            T.StructField("recall", T.DoubleType(), False),
            T.StructField("area_under_roc", T.DoubleType(), True),
            T.StructField("area_under_pr", T.DoubleType(), True),
            T.StructField("brier_score", T.DoubleType(), False),
            T.StructField("top_5_capture_rate", T.DoubleType(), False),
            T.StructField("top_5_lift", T.DoubleType(), False),
            T.StructField("top_10_capture_rate", T.DoubleType(), False),
            T.StructField("top_10_lift", T.DoubleType(), False),
            T.StructField("high_cost_threshold_train_only", T.DoubleType(), False),
            T.StructField("decision_threshold_validation_tuned", T.DoubleType(), False),
            T.StructField("decision_threshold_from_tuning_split", T.DoubleType(), False),
            T.StructField("shared_split_version", T.StringType(), False),
            T.StructField("processed_at_utc", T.TimestampType(), True),
        ]
    )

    rows = []
    for metric in metrics:
        rows.append(
            {
                "run_id": run_id,
                "split_name": metric["split_name"],
                "row_count": metric["row_count"],
                "positive_rate": metric["positive_rate"],
                "accuracy": metric["accuracy"],
                "precision": metric["precision"],
                "recall": metric["recall"],
                "area_under_roc": metric["area_under_roc"],
                "area_under_pr": metric["area_under_pr"],
                "brier_score": metric["brier_score"],
                "top_5_capture_rate": metric["top_5_capture_rate"],
                "top_5_lift": metric["top_5_lift"],
                "top_10_capture_rate": metric["top_10_capture_rate"],
                "top_10_lift": metric["top_10_lift"],
                "high_cost_threshold_train_only": label_threshold,
                "decision_threshold_validation_tuned": decision_threshold,
                "decision_threshold_from_tuning_split": decision_threshold,
                "shared_split_version": SHARED_SPLIT_VERSION,
                "processed_at_utc": None,
            }
        )

    return spark.createDataFrame(rows, schema=schema).withColumn("processed_at_utc", F.current_timestamp())


def write_audit_table(df: DataFrame) -> None:
    (
        df.write.format("delta")
        .mode("append")
        .saveAsTable(f"{MODEL_DATABASE}.{TRAINING_AUDIT_TABLE}")
    )


def create_topk_curve_df(curve_rows: list[dict[str, float | str]], run_id: str) -> DataFrame:
    schema = T.StructType(
        [
            T.StructField("run_id", T.StringType(), False),
            T.StructField("model_name", T.StringType(), False),
            T.StructField("split_name", T.StringType(), False),
            T.StructField("selected_fraction", T.DoubleType(), False),
            T.StructField("capture_rate", T.DoubleType(), False),
            T.StructField("lift", T.DoubleType(), False),
            T.StructField("processed_at_utc", T.TimestampType(), True),
        ]
    )
    rows = [{"run_id": run_id, **row, "processed_at_utc": None} for row in curve_rows]
    return spark.createDataFrame(rows, schema=schema).withColumn("processed_at_utc", F.current_timestamp())


def build_topk_curve_rows(y_true, y_score, split_name: str) -> list[dict[str, float | str]]:
    rows = []
    for fraction in np.linspace(0.01, 0.2, 20):
        capture, lift = top_k_capture_and_lift(y_true, y_score, float(fraction))
        rows.append(
            {
                "model_name": MODEL_NAME,
                "split_name": split_name,
                "selected_fraction": float(fraction),
                "capture_rate": float(capture),
                "lift": float(lift),
            }
        )
    return rows


def append_topk_curve(df: DataFrame) -> None:
    (
        df.write.format("delta")
        .mode("append")
        .saveAsTable(f"{MODEL_DATABASE}.{TOPK_CURVE_TABLE}")
    )


def create_prediction_score_df(score_rows: list[dict[str, object]], run_id: str) -> DataFrame:
    schema = T.StructType(
        [
            T.StructField("run_id", T.StringType(), False),
            T.StructField("model_name", T.StringType(), False),
            T.StructField("split_name", T.StringType(), False),
            T.StructField("bene_id", T.StringType(), False),
            T.StructField("year", T.IntegerType(), False),
            T.StructField("target_year", T.IntegerType(), False),
            T.StructField("target_annual_claim_cost", T.DoubleType(), False),
            T.StructField("label", T.DoubleType(), False),
            T.StructField("predicted_probability", T.DoubleType(), False),
            T.StructField("predicted_label", T.IntegerType(), False),
            T.StructField("processed_at_utc", T.TimestampType(), True),
        ]
    )
    rows = [{"run_id": run_id, "model_name": MODEL_NAME, **row, "processed_at_utc": None} for row in score_rows]
    return spark.createDataFrame(rows, schema=schema).withColumn("processed_at_utc", F.current_timestamp())


def append_prediction_scores(df: DataFrame) -> None:
    (
        df.write.format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(f"{MODEL_DATABASE}.{PREDICTION_SCORE_TABLE}")
    )


def build_prediction_score_rows(pdf: pd.DataFrame, y_score, y_pred, split_name: str) -> list[dict[str, object]]:
    scored = pdf[["bene_id", "year", "target_year", "target_annual_claim_cost", "label"]].copy()
    scored["predicted_probability"] = y_score
    scored["predicted_label"] = y_pred
    scored["split_name"] = split_name
    return scored.to_dict("records")


def main() -> None:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {MODEL_DATABASE}")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)

    gold_df = read_gold()
    modeling_df = build_modeling_frame(gold_df)
    validate_required_columns(modeling_df, REQUIRED_MODELING_COLUMNS, "modeling_df")
    validate_unique_keys(modeling_df, ["bene_id", "year", "target_year"], "modeling_df")
    train_df, validation_df, test_df = split_gold_by_time(modeling_df)
    train_df, validation_df, test_df, threshold = add_training_target(train_df, validation_df, test_df)

    train_pdf = to_pandas_features(train_df)
    validation_pdf = to_pandas_features(validation_df)
    test_pdf = to_pandas_features(test_df)

    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x_train = train_pdf[feature_columns]
    y_train = train_pdf["label"].astype(int)
    x_validation = validation_pdf[feature_columns]
    y_validation = validation_pdf["label"].astype(int)
    x_test = test_pdf[feature_columns]
    y_test = test_pdf["label"].astype(int)
    validate_label_distribution(y_train, "train", require_both_classes=True)
    validate_label_distribution(y_validation, "validation", require_both_classes=False)
    validate_label_distribution(y_test, "test", require_both_classes=False)
    scale_pos_weight = compute_scale_pos_weight(y_train)

    with mlflow.start_run(run_name="xgboost") as run:
        pipeline = fit_final_pipeline_with_early_stopping(
            scale_pos_weight=scale_pos_weight,
            x_train=x_train,
            y_train=y_train,
            x_validation=x_validation,
            y_validation=y_validation,
        )
        calibrated_pipeline = calibrate_prefit_model(pipeline, x_validation, y_validation)

        validation_scores = calibrated_pipeline.predict_proba(x_validation)[:, 1]
        decision_threshold = FIXED_DECISION_THRESHOLD

        train_scores = calibrated_pipeline.predict_proba(x_train)[:, 1]
        test_scores = calibrated_pipeline.predict_proba(x_test)[:, 1]

        train_predictions = (train_scores >= decision_threshold).astype(int)
        validation_predictions = (validation_scores >= decision_threshold).astype(int)
        test_predictions = (test_scores >= decision_threshold).astype(int)

        metrics = [
            evaluate_predictions(y_train, train_scores, train_predictions, "train"),
            evaluate_predictions(y_validation, validation_scores, validation_predictions, "validation"),
            evaluate_predictions(y_test, test_scores, test_predictions, "test"),
        ]

        curve_rows = []
        curve_rows.extend(build_topk_curve_rows(y_train, train_scores, "train"))
        curve_rows.extend(build_topk_curve_rows(y_validation, validation_scores, "validation"))
        curve_rows.extend(build_topk_curve_rows(y_test, test_scores, "test"))
        score_rows = []
        score_rows.extend(build_prediction_score_rows(train_pdf, train_scores, train_predictions, "train"))
        score_rows.extend(build_prediction_score_rows(validation_pdf, validation_scores, validation_predictions, "validation"))
        score_rows.extend(build_prediction_score_rows(test_pdf, test_scores, test_predictions, "test"))

        mlflow.log_param("model_family", "xgboost")
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("methodological_reference", "Chen and Guestrin 2016 XGBoost")
        mlflow.log_param("training_objective", "regularized_second_order_additive_tree_ensemble")
        mlflow.log_param("train_target_quantile", 0.9)
        mlflow.log_param("high_cost_threshold_train_only", threshold)
        mlflow.log_param("decision_threshold", decision_threshold)
        mlflow.log_param("decision_threshold_policy", "fixed_0_5_probability_cutoff")
        mlflow.log_param("threshold_selection_split", "none")
        mlflow.log_param("final_evaluation_split", "test")
        mlflow.log_param("split_strategy", SPLIT_STRATEGY)
        mlflow.log_param("split_stratification", "target_year_temporal_holdout")
        mlflow.log_param("shared_split_version", SHARED_SPLIT_VERSION)
        mlflow.log_param("gold_table", f"{GOLD_DATABASE}.{GOLD_TABLE_NAME}")
        mlflow.log_param("feature_timing_frame", "current_year_features_predict_next_year_target")
        mlflow.log_param("utilization_feature_timing", "prior_year_relative_to_target_year")
        mlflow.log_param("max_pandas_rows_per_split", MAX_PANDAS_ROWS_PER_SPLIT)
        mlflow.log_param("hyperparameter_selection", "fixed_paper_aligned_baseline_plus_validation_early_stopping")
        mlflow.log_param("primary_training_metric", "logloss")
        mlflow.log_param("secondary_reporting_metrics", "roc_auc,average_precision,top_k_capture,lift")
        mlflow.log_param("target_definition", "predict_next_year_high_cost_within_target_year_top_decile")
        mlflow.log_param("numeric_features", ",".join(NUMERIC_FEATURES))
        mlflow.log_param("categorical_features", ",".join(CATEGORICAL_FEATURES))
        mlflow.log_param("numeric_missing_value_handling", "xgboost_native_missing_direction")
        mlflow.log_param("categorical_encoding", "one_hot_without_missing_imputation")
        mlflow.log_param("class_imbalance_strategy", "training_split_scale_pos_weight")
        mlflow.log_param("training_scale_pos_weight", scale_pos_weight)
        mlflow.log_param("probability_calibration", CALIBRATION_METHOD)
        mlflow.log_param(
            "calibration_method_detail",
            "platt_sigmoid" if CALIBRATION_METHOD == "sigmoid" else "isotonic_regression",
        )
        mlflow.log_param("calibration_split", "validation")
        mlflow.log_param("early_stopping_rounds", EARLY_STOPPING_ROUNDS)
        mlflow.log_param("tree_method_interpretation", "hist_approximate_split_finding_for_scale")
        mlflow.log_param("best_classifier_n_estimators", int(pipeline.named_steps["classifier"].best_iteration + 1))
        mlflow.log_param(
            "best_classifier_learning_rate",
            float(pipeline.named_steps["classifier"].get_xgb_params()["learning_rate"]),
        )
        mlflow.log_param(
            "best_classifier_max_depth",
            int(pipeline.named_steps["classifier"].get_xgb_params()["max_depth"]),
        )
        mlflow.log_param(
            "best_classifier_min_child_weight",
            int(pipeline.named_steps["classifier"].get_xgb_params()["min_child_weight"]),
        )
        mlflow.log_param(
            "best_classifier_subsample",
            float(pipeline.named_steps["classifier"].get_xgb_params()["subsample"]),
        )
        mlflow.log_param(
            "best_classifier_colsample_bytree",
            float(pipeline.named_steps["classifier"].get_xgb_params()["colsample_bytree"]),
        )
        mlflow.log_param(
            "best_classifier_gamma",
            float(pipeline.named_steps["classifier"].get_xgb_params()["gamma"]),
        )
        mlflow.log_param(
            "best_classifier_reg_lambda",
            float(pipeline.named_steps["classifier"].get_xgb_params()["reg_lambda"]),
        )
        mlflow.log_param(
            "best_classifier_reg_alpha",
            float(pipeline.named_steps["classifier"].get_xgb_params()["reg_alpha"]),
        )
        mlflow.log_param(
            "best_classifier_scale_pos_weight",
            float(pipeline.named_steps["classifier"].get_xgb_params()["scale_pos_weight"]),
        )
        mlflow.log_metric(
            "best_classifier_best_iteration",
            float(pipeline.named_steps["classifier"].best_iteration + 1),
        )

        for metric in metrics:
            prefix = metric["split_name"]
            for metric_name, metric_value in metric.items():
                if metric_name == "split_name" or metric_value is None:
                    continue
                mlflow.log_metric(f"{prefix}_{metric_name}", metric_value)

        signature = infer_signature(x_train.head(20), calibrated_pipeline.predict_proba(x_train.head(20)))
        mlflow.sklearn.log_model(calibrated_pipeline, artifact_path="model", signature=signature)

        audit_df = create_audit_df(metrics, threshold, decision_threshold, run.info.run_id)
        write_audit_table(audit_df)

        curve_df = create_topk_curve_df(curve_rows, run.info.run_id)
        append_topk_curve(curve_df)
        append_prediction_scores(create_prediction_score_df(score_rows, run.info.run_id))

        print(f"wrote audit table {MODEL_DATABASE}.{TRAINING_AUDIT_TABLE}")
        display(audit_df.orderBy("split_name"))
        display(curve_df.filter(F.col("split_name") == "test").orderBy("selected_fraction"))


# COMMAND ----------

main()
