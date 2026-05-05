# Databricks notebook source
# MAGIC %md
# MAGIC # Train Tree Baseline
# MAGIC
# MAGIC Trains a nonlinear random-forest baseline on the beneficiary-level next-year prediction task and logs ranking plus top-k targeting metrics.

# COMMAND ----------

from __future__ import annotations

import math
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from databricks.modeling_utils import apply_threshold, compute_training_only_threshold, reject_target_leakage


GOLD_DATABASE = os.environ.get("GOLD_DATABASE", "default")
MODEL_DATABASE = os.environ.get("MODEL_DATABASE", GOLD_DATABASE)
GOLD_TABLE_NAME = "gold_beneficiary_year_features"
TRAINING_AUDIT_TABLE = "tree_training_audit"
TOPK_CURVE_TABLE = "model_topk_curve_points"
PREDICTION_SCORE_TABLE = "model_prediction_scores"
MODEL_NAME = "random_forest"
RANDOM_SEED = 42
MLFLOW_EXPERIMENT_PATH = os.environ.get(
    "MLFLOW_EXPERIMENT_PATH",
    "/Shared/Project2HighCostClaimClassifier_Experiment",
)
# Final locked comparison profile. Keep the grid small enough for Databricks
# job-task timeouts; this is a comparison baseline, not an exhaustive RF search.
CV_FOLDS = int(os.environ.get("RF_CV_FOLDS", "3"))
RF_N_ESTIMATORS = int(os.environ.get("RF_N_ESTIMATORS", "150"))
TARGET_QUANTILE = 0.9
MAX_DRIVER_ROWS = int(os.environ.get("MAX_DRIVER_ROWS", "1000000"))
SPLIT_STRATEGY = "temporal_target_year_holdout"
ALLOW_DELTA_MERGE_SCHEMA = os.environ.get("ALLOW_DELTA_MERGE_SCHEMA", "false").lower() == "true"
SHARED_SPLIT_VERSION = "xxhash64_bene_id_mod_100_v1"
VALIDATION_BUCKET_CUTOFF = 15
CALIBRATION_METHOD = os.environ.get("RF_CALIBRATION_METHOD", "sigmoid")

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


def read_gold() -> DataFrame:
    table_name = f"{GOLD_DATABASE}.{GOLD_TABLE_NAME}"
    try:
        return spark.table(table_name)
    except Exception as exc:
        raise ValueError(f"Gold table not found: {table_name}. Run 03_gold.py before training this model.") from exc


def validate_gold_frame(df: DataFrame) -> None:
    required_columns = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["bene_id", "year", "annual_claim_cost"])
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"Gold table is missing required columns: {missing_columns}")

    null_key_count = df.filter(F.col("bene_id").isNull() | F.col("year").isNull()).limit(1).count()
    if null_key_count:
        raise ValueError("Gold table contains null bene_id or year values.")

    null_cost_count = df.filter(F.col("annual_claim_cost").isNull()).limit(1).count()
    if null_cost_count:
        raise ValueError("Gold table contains null annual_claim_cost values.")

    duplicate_count = df.groupBy("bene_id", "year").count().filter(F.col("count") > 1).limit(1).count()
    if duplicate_count:
        raise ValueError("Gold table must have at most one row per bene_id/year before prospective joining.")


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
    return training_pool.join(train_ids, "bene_id", "inner"), training_pool.join(validation_ids, "bene_id", "inner"), test_df


def add_training_target(
    train_df: DataFrame,
    validation_df: DataFrame,
    test_df: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame, float]:
    threshold = compute_training_only_threshold(train_df, TARGET_QUANTILE)
    return apply_threshold(train_df, threshold), apply_threshold(validation_df, threshold), apply_threshold(test_df, threshold), threshold


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="constant", fill_value=0.0), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=RF_N_ESTIMATORS,
                    min_samples_leaf=10,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )


def build_cv_search() -> GridSearchCV:
    return GridSearchCV(
        estimator=build_pipeline(),
        param_grid={
            "classifier__max_depth": [8, 12],
            "classifier__min_samples_leaf": [10, 20],
            "classifier__max_features": ["sqrt"],
        },
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )


def calibrate_prefit_model(model: Pipeline, x_calibration: pd.DataFrame, y_calibration: pd.Series) -> CalibratedClassifierCV:
    if CALIBRATION_METHOD not in {"sigmoid", "isotonic"}:
        raise ValueError("RF_CALIBRATION_METHOD must be either 'sigmoid' for Platt scaling or 'isotonic'.")
    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method=CALIBRATION_METHOD,
        cv="prefit",
    )
    calibrated_model.fit(x_calibration, y_calibration)
    return calibrated_model


def to_pandas_features(df: DataFrame) -> pd.DataFrame:
    selected_df = df.select(
        "bene_id",
        "year",
        "target_year",
        "target_annual_claim_cost",
        *(NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["label"]),
    )
    row_count = selected_df.count()
    if row_count > MAX_DRIVER_ROWS:
        raise ValueError(
            f"Refusing to collect {row_count} rows to the driver. "
            f"Increase MAX_DRIVER_ROWS if this run is intentionally driver-local."
        )
    return selected_df.toPandas()


def choose_decision_threshold(y_true, y_score) -> float:
    score_quantiles = np.quantile(np.asarray(y_score), np.linspace(0.0, 1.0, 101))
    candidate_thresholds = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), score_quantiles]))
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidate_thresholds:
        predictions = (y_score >= threshold).astype(int)
        score = f1_score(y_true, predictions, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def metric_auc(y_true, y_score) -> tuple[float, float]:
    if pd.Series(y_true).nunique(dropna=False) < 2:
        return math.nan, math.nan
    return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))


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


def evaluate_predictions(y_true, y_score, y_pred, split_name: str) -> dict[str, float | int | str]:
    capture_5, lift_5 = top_k_capture_and_lift(y_true, y_score, 0.05)
    capture_10, lift_10 = top_k_capture_and_lift(y_true, y_score, 0.10)
    area_under_roc, area_under_pr = metric_auc(y_true, y_score)
    return {
        "split_name": split_name,
        "row_count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "area_under_roc": area_under_roc,
        "area_under_pr": area_under_pr,
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "positive_rate": float(y_true.mean()),
        "top_5_capture_rate": float(capture_5),
        "top_5_lift": float(lift_5),
        "top_10_capture_rate": float(capture_10),
        "top_10_lift": float(lift_10),
    }


def create_audit_df(
    metrics: list[dict[str, float | int | str]],
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
            T.StructField("area_under_roc", T.DoubleType(), False),
            T.StructField("area_under_pr", T.DoubleType(), False),
            T.StructField("brier_score", T.DoubleType(), False),
            T.StructField("top_5_capture_rate", T.DoubleType(), False),
            T.StructField("top_5_lift", T.DoubleType(), False),
            T.StructField("top_10_capture_rate", T.DoubleType(), False),
            T.StructField("top_10_lift", T.DoubleType(), False),
            T.StructField("high_cost_threshold_train_only", T.DoubleType(), False),
            T.StructField("decision_threshold_validation_tuned", T.DoubleType(), False),
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
                "shared_split_version": SHARED_SPLIT_VERSION,
                "processed_at_utc": None,
            }
        )

    return spark.createDataFrame(rows, schema=schema).withColumn("processed_at_utc", F.current_timestamp())


def write_audit_table(df: DataFrame) -> None:
    append_delta_table(df, f"{MODEL_DATABASE}.{TRAINING_AUDIT_TABLE}")


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


def write_topk_curve_table(df: DataFrame) -> None:
    append_delta_table(df, f"{MODEL_DATABASE}.{TOPK_CURVE_TABLE}")


def append_delta_table(df: DataFrame, table_name: str) -> None:
    writer = df.write.format("delta").mode("append")
    if ALLOW_DELTA_MERGE_SCHEMA:
        writer = writer.option("mergeSchema", "true")
    writer.saveAsTable(table_name)


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


def build_prediction_score_rows(pdf: pd.DataFrame, y_score, y_pred, split_name: str) -> list[dict[str, object]]:
    scored = pdf[["bene_id", "year", "target_year", "target_annual_claim_cost", "label"]].copy()
    scored["predicted_probability"] = y_score
    scored["predicted_label"] = y_pred
    scored["split_name"] = split_name
    return scored.to_dict("records")


def validate_labeled_split(pdf: pd.DataFrame, split_name: str, require_two_classes: bool = False) -> None:
    row_count = len(pdf)
    if row_count == 0:
        raise ValueError(f"{split_name} split is empty after beneficiary-level splitting.")

    class_count = pdf["label"].nunique(dropna=False)
    if require_two_classes and class_count < 2:
        raise ValueError(f"{split_name} split must contain both target classes before model fitting.")


def log_metric_if_finite(name: str, value: float | int) -> None:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return
    mlflow.log_metric(name, value)


def main() -> None:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {MODEL_DATABASE}")

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks")

    source_gold_df = read_gold()
    validate_gold_frame(source_gold_df)
    gold_df = build_modeling_frame(source_gold_df)
    train_df, validation_df, test_df = split_gold_by_time(gold_df)
    train_df, validation_df, test_df, threshold = add_training_target(train_df, validation_df, test_df)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)

    with mlflow.start_run(run_name="tree_baseline_random_forest") as run:
        reject_target_leakage(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        train_pdf = to_pandas_features(train_df)
        validation_pdf = to_pandas_features(validation_df)
        test_pdf = to_pandas_features(test_df)
        validate_labeled_split(train_pdf, "train", require_two_classes=True)
        validate_labeled_split(validation_pdf, "validation", require_two_classes=True)
        validate_labeled_split(test_pdf, "test")

        cv_search = build_cv_search()
        cv_search.fit(
            train_pdf[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
            train_pdf["label"],
        )
        model = cv_search.best_estimator_
        calibrated_model = calibrate_prefit_model(
            model,
            validation_pdf[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
            validation_pdf["label"],
        )

        train_scores = calibrated_model.predict_proba(train_pdf[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
        validation_scores = calibrated_model.predict_proba(validation_pdf[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
        test_scores = calibrated_model.predict_proba(test_pdf[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]

        decision_threshold = choose_decision_threshold(validation_pdf["label"], validation_scores)

        train_predictions = (train_scores >= decision_threshold).astype(int)
        validation_predictions = (validation_scores >= decision_threshold).astype(int)
        test_predictions = (test_scores >= decision_threshold).astype(int)

        metrics = [
            evaluate_predictions(train_pdf["label"], train_scores, train_predictions, "train"),
            evaluate_predictions(validation_pdf["label"], validation_scores, validation_predictions, "validation"),
            evaluate_predictions(test_pdf["label"], test_scores, test_predictions, "test"),
        ]
        curve_rows = []
        curve_rows.extend(build_topk_curve_rows(train_pdf["label"], train_scores, "train"))
        curve_rows.extend(build_topk_curve_rows(validation_pdf["label"], validation_scores, "validation"))
        curve_rows.extend(build_topk_curve_rows(test_pdf["label"], test_scores, "test"))
        score_rows = []
        score_rows.extend(build_prediction_score_rows(train_pdf, train_scores, train_predictions, "train"))
        score_rows.extend(build_prediction_score_rows(validation_pdf, validation_scores, validation_predictions, "validation"))
        score_rows.extend(build_prediction_score_rows(test_pdf, test_scores, test_predictions, "test"))

        mlflow.log_param("model_family", "random_forest")
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("train_target_quantile", TARGET_QUANTILE)
        mlflow.log_param("high_cost_threshold_train_only", threshold)
        mlflow.log_param("decision_threshold_from_tuning_split", decision_threshold)
        mlflow.log_param("threshold_selection_split", "validation")
        mlflow.log_param("probability_calibration", CALIBRATION_METHOD)
        mlflow.log_param(
            "calibration_method_detail",
            "platt_sigmoid" if CALIBRATION_METHOD == "sigmoid" else "isotonic_regression",
        )
        mlflow.log_param("calibration_split", "validation")
        mlflow.log_param("final_evaluation_split", "test")
        mlflow.log_param("split_strategy", SPLIT_STRATEGY)
        mlflow.log_param("gold_table", f"{GOLD_DATABASE}.{GOLD_TABLE_NAME}")
        mlflow.log_param("split_stratification_basis", "target_year_temporal_holdout")
        mlflow.log_param("shared_split_version", SHARED_SPLIT_VERSION)
        mlflow.log_param("feature_timing_frame", "current_year_features_predict_next_year_target")
        mlflow.log_param("utilization_feature_timing", "prior_year_relative_to_target_year")
        mlflow.log_param("training_runtime", "driver_local_sklearn_after_spark_to_pandas")
        mlflow.log_param("max_driver_rows", MAX_DRIVER_ROWS)
        mlflow.log_param("delta_merge_schema_enabled", ALLOW_DELTA_MERGE_SCHEMA)
        mlflow.log_param("write_transaction_scope", "independent_delta_appends")
        mlflow.log_param("hyperparameter_selection", "5_fold_cv_on_training_split")
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("cv_scoring_metric", "average_precision")
        mlflow.log_param("target_definition", "predict_next_year_high_cost_within_target_year_top_decile")
        mlflow.log_param("numeric_features", ",".join(NUMERIC_FEATURES))
        mlflow.log_param("categorical_features", ",".join(CATEGORICAL_FEATURES))
        mlflow.log_param("best_classifier_max_depth", str(cv_search.best_params_["classifier__max_depth"]))
        mlflow.log_param("best_classifier_min_samples_leaf", int(cv_search.best_params_["classifier__min_samples_leaf"]))
        mlflow.log_param("best_classifier_max_features", str(cv_search.best_params_["classifier__max_features"]))
        mlflow.log_metric("cv_best_average_precision", float(cv_search.best_score_))

        for metric in metrics:
            prefix = metric["split_name"]
            log_metric_if_finite(f"{prefix}_row_count", metric["row_count"])
            log_metric_if_finite(f"{prefix}_positive_rate", metric["positive_rate"])
            log_metric_if_finite(f"{prefix}_accuracy", metric["accuracy"])
            log_metric_if_finite(f"{prefix}_precision", metric["precision"])
            log_metric_if_finite(f"{prefix}_recall", metric["recall"])
            log_metric_if_finite(f"{prefix}_area_under_roc", metric["area_under_roc"])
            log_metric_if_finite(f"{prefix}_area_under_pr", metric["area_under_pr"])
            log_metric_if_finite(f"{prefix}_brier_score", metric["brier_score"])
            log_metric_if_finite(f"{prefix}_top_5_capture_rate", metric["top_5_capture_rate"])
            log_metric_if_finite(f"{prefix}_top_5_lift", metric["top_5_lift"])
            log_metric_if_finite(f"{prefix}_top_10_capture_rate", metric["top_10_capture_rate"])
            log_metric_if_finite(f"{prefix}_top_10_lift", metric["top_10_lift"])

        signature = infer_signature(
            train_pdf[NUMERIC_FEATURES + CATEGORICAL_FEATURES].head(100),
            calibrated_model.predict_proba(train_pdf[NUMERIC_FEATURES + CATEGORICAL_FEATURES].head(100)),
        )
        mlflow.sklearn.log_model(calibrated_model, "model", signature=signature)

        audit_df = create_audit_df(metrics, threshold, decision_threshold, run.info.run_id)
        write_audit_table(audit_df)
        write_topk_curve_table(create_topk_curve_df(curve_rows, run.info.run_id))
        append_delta_table(create_prediction_score_df(score_rows, run.info.run_id), f"{MODEL_DATABASE}.{PREDICTION_SCORE_TABLE}")

        for metric in metrics:
            print(
                f"{metric['split_name']}: row_count={metric['row_count']} "
                f"positive_rate={metric['positive_rate']:.4f} "
                f"accuracy={metric['accuracy']:.4f} "
                f"precision={metric['precision']:.4f} "
                f"recall={metric['recall']:.4f} "
                f"auc_roc={metric['area_under_roc']:.4f} "
                f"auc_pr={metric['area_under_pr']:.4f} "
                f"brier={metric['brier_score']:.4f} "
                f"top_5_capture={metric['top_5_capture_rate']:.4f} "
                f"top_5_lift={metric['top_5_lift']:.4f} "
                f"top_10_capture={metric['top_10_capture_rate']:.4f} "
                f"top_10_lift={metric['top_10_lift']:.4f}"
            )
        print(f"cv_best_params={cv_search.best_params_}")
        print(f"cv_best_average_precision={cv_search.best_score_:.4f}")
        print(f"training audit written to {MODEL_DATABASE}.{TRAINING_AUDIT_TABLE}")
        print(f"top-k curve written to {MODEL_DATABASE}.{TOPK_CURVE_TABLE}")
        print(f"mlflow_run_id={run.info.run_id}")


# COMMAND ----------

main()
