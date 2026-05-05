# Databricks notebook source
# MAGIC %md
# MAGIC # Train Logistic Regression Baseline
# MAGIC
# MAGIC Recomputes the top-decile target from the training split only, trains a logistic regression baseline, and logs metrics and the fitted pipeline to MLflow.

# COMMAND ----------

from __future__ import annotations

import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from databricks.modeling_utils import apply_threshold, compute_training_only_threshold, reject_target_leakage


GOLD_DATABASE = os.environ.get("GOLD_DATABASE", "default")
MODEL_DATABASE = os.environ.get("MODEL_DATABASE", GOLD_DATABASE)
GOLD_TABLE_NAME = "gold_beneficiary_year_features"
TRAINING_AUDIT_TABLE = "logreg_training_audit"
TOPK_CURVE_TABLE = "model_topk_curve_points"
PREDICTION_SCORE_TABLE = "model_prediction_scores"
MODEL_NAME = "logistic_regression"
RANDOM_SEED = 42
MLFLOW_EXPERIMENT_PATH = os.environ.get(
    "MLFLOW_EXPERIMENT_PATH",
    "/Shared/Project2HighCostClaimClassifier_Experiment",
)
# Final locked comparison profile.
CV_FOLDS = 5
TARGET_QUANTILE = 0.9
SPLIT_STRATEGY = "temporal_target_year_holdout"
SHARED_SPLIT_VERSION = "xxhash64_bene_id_mod_100_v1"
VALIDATION_BUCKET_CUTOFF = 15

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
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)),
        ]
    )


def build_cv_search() -> GridSearchCV:
    return GridSearchCV(
        estimator=build_pipeline(),
        param_grid={
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs"],
        },
        scoring={
            "average_precision": "average_precision",
            "roc_auc": "roc_auc",
            "default_threshold_accuracy": "accuracy",
        },
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
        n_jobs=-1,
        refit="average_precision",
        return_train_score=True,
    )


def to_pandas_features(df: DataFrame) -> pd.DataFrame:
    return df.select(
        "bene_id",
        "year",
        "target_year",
        "target_annual_claim_cost",
        *(NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["label"]),
    ).toPandas()


def choose_decision_threshold(y_true, y_score) -> float:
    candidate_thresholds = np.linspace(0.1, 0.9, 17)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidate_thresholds:
        predictions = (y_score >= threshold).astype(int)
        score = f1_score(y_true, predictions, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def top_k_capture_and_lift(y_true, y_score, top_fraction: float) -> tuple[float, float]:
    top_n = max(1, int(np.ceil(len(y_true) * top_fraction)))
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
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_true,
        y_pred,
        labels=[0.0, 1.0],
    ).ravel()
    accuracy = float(accuracy_score(y_true, y_pred))
    return {
        "split_name": split_name,
        "row_count": int(len(y_true)),
        "accuracy": accuracy,
        "test_error": float(1.0 - accuracy),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "area_under_roc": float(roc_auc_score(y_true, y_score)),
        "area_under_pr": float(average_precision_score(y_true, y_score)),
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
            T.StructField("test_error", T.DoubleType(), False),
            T.StructField("true_negative", T.LongType(), False),
            T.StructField("false_positive", T.LongType(), False),
            T.StructField("false_negative", T.LongType(), False),
            T.StructField("true_positive", T.LongType(), False),
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
                "test_error": metric["test_error"],
                "true_negative": metric["true_negative"],
                "false_positive": metric["false_positive"],
                "false_negative": metric["false_negative"],
                "true_positive": metric["true_positive"],
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
        .option("mergeSchema", "true")
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


def write_topk_curve_table(df: DataFrame) -> None:
    (
        df.write.format("delta")
        .mode("append")
        .option("mergeSchema", "true")
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


def write_prediction_score_table(df: DataFrame) -> None:
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

    gold_df = build_modeling_frame(read_gold())
    train_df, validation_df, test_df = split_gold_by_time(gold_df)
    train_df, validation_df, test_df, threshold = add_training_target(train_df, validation_df, test_df)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)

    with mlflow.start_run(run_name="logreg_baseline") as run:
        reject_target_leakage(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
        train_pdf = to_pandas_features(train_df)
        validation_pdf = to_pandas_features(validation_df)
        test_pdf = to_pandas_features(test_df)

        cv_search = build_cv_search()
        cv_search.fit(
            train_pdf[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
            train_pdf["label"],
        )
        model = cv_search.best_estimator_

        train_scores = model.predict_proba(train_pdf[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
        validation_scores = model.predict_proba(validation_pdf[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
        test_scores = model.predict_proba(test_pdf[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]

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

        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_param("train_target_quantile", 0.9)
        mlflow.log_param("high_cost_threshold_train_only", threshold)
        mlflow.log_param("decision_threshold_from_tuning_split", decision_threshold)
        mlflow.log_param("threshold_selection_split", "validation")
        mlflow.log_param("final_evaluation_split", "test")
        mlflow.log_param("split_strategy", SPLIT_STRATEGY)
        mlflow.log_param("shared_split_version", SHARED_SPLIT_VERSION)
        mlflow.log_param("feature_timing_frame", "current_year_features_predict_next_year_target")
        mlflow.log_param("utilization_feature_timing", "prior_year_relative_to_target_year")
        mlflow.log_param("hyperparameter_selection", "5_fold_cv_on_training_split")
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("cv_scoring_metric", "average_precision")
        mlflow.log_param("cv_secondary_metrics", "roc_auc,default_threshold_accuracy")
        mlflow.log_param("target_definition", "predict_next_year_high_cost_within_target_year_top_decile")
        mlflow.log_param("numeric_features", ",".join(NUMERIC_FEATURES))
        mlflow.log_param("categorical_features", ",".join(CATEGORICAL_FEATURES))
        mlflow.log_param("best_classifier_C", float(cv_search.best_params_["classifier__C"]))
        best_cv_index = cv_search.best_index_
        mlflow.log_metric("cv_best_average_precision", float(cv_search.best_score_))
        mlflow.log_metric("cv_best_roc_auc", float(cv_search.cv_results_["mean_test_roc_auc"][best_cv_index]))
        mlflow.log_metric(
            "cv_best_default_threshold_accuracy",
            float(cv_search.cv_results_["mean_test_default_threshold_accuracy"][best_cv_index]),
        )

        for metric in metrics:
            prefix = metric["split_name"]
            mlflow.log_metric(f"{prefix}_row_count", metric["row_count"])
            mlflow.log_metric(f"{prefix}_positive_rate", metric["positive_rate"])
            mlflow.log_metric(f"{prefix}_accuracy", metric["accuracy"])
            mlflow.log_metric(f"{prefix}_test_error", metric["test_error"])
            mlflow.log_metric(f"{prefix}_true_negative", metric["true_negative"])
            mlflow.log_metric(f"{prefix}_false_positive", metric["false_positive"])
            mlflow.log_metric(f"{prefix}_false_negative", metric["false_negative"])
            mlflow.log_metric(f"{prefix}_true_positive", metric["true_positive"])
            mlflow.log_metric(f"{prefix}_precision", metric["precision"])
            mlflow.log_metric(f"{prefix}_recall", metric["recall"])
            mlflow.log_metric(f"{prefix}_area_under_roc", metric["area_under_roc"])
            mlflow.log_metric(f"{prefix}_area_under_pr", metric["area_under_pr"])
            mlflow.log_metric(f"{prefix}_brier_score", metric["brier_score"])
            mlflow.log_metric(f"{prefix}_top_5_capture_rate", metric["top_5_capture_rate"])
            mlflow.log_metric(f"{prefix}_top_5_lift", metric["top_5_lift"])
            mlflow.log_metric(f"{prefix}_top_10_capture_rate", metric["top_10_capture_rate"])
            mlflow.log_metric(f"{prefix}_top_10_lift", metric["top_10_lift"])

        signature = infer_signature(
            train_pdf[NUMERIC_FEATURES + CATEGORICAL_FEATURES].head(100),
            model.predict_proba(train_pdf[NUMERIC_FEATURES + CATEGORICAL_FEATURES].head(100)),
        )
        mlflow.sklearn.log_model(model, "model", signature=signature)

        audit_df = create_audit_df(metrics, threshold, decision_threshold, run.info.run_id)
        write_audit_table(audit_df)
        write_topk_curve_table(create_topk_curve_df(curve_rows, run.info.run_id))
        write_prediction_score_table(create_prediction_score_df(score_rows, run.info.run_id))

        for metric in metrics:
            print(
                f"{metric['split_name']}: row_count={metric['row_count']} "
                f"positive_rate={metric['positive_rate']:.4f} "
                f"accuracy={metric['accuracy']:.4f} "
                f"test_error={metric['test_error']:.4f} "
                f"tn={metric['true_negative']} "
                f"fp={metric['false_positive']} "
                f"fn={metric['false_negative']} "
                f"tp={metric['true_positive']} "
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
        print(f"cv_best_roc_auc={cv_search.cv_results_['mean_test_roc_auc'][best_cv_index]:.4f}")
        print(
            "cv_best_default_threshold_accuracy="
            f"{cv_search.cv_results_['mean_test_default_threshold_accuracy'][best_cv_index]:.4f}"
        )
        print(
            "validation split is used for decision-threshold tuning; "
            "test split is the final held-out performance estimate"
        )
        print(f"training audit written to {MODEL_DATABASE}.{TRAINING_AUDIT_TABLE}")
        print(f"mlflow_run_id={run.info.run_id}")


# COMMAND ----------

main()
