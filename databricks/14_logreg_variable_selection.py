# Databricks notebook source
# MAGIC %md
# MAGIC # Logistic Regression Variable Selection
# MAGIC
# MAGIC Statistics-facing logistic benchmark with training-only centering, VIF-screened candidate terms,
# MAGIC backward AIC, hierarchy enforcement, nested likelihood-ratio tests, final VIF output, and odds ratios.

# COMMAND ----------

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from scipy.stats import chi2, norm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pyspark.sql import DataFrame
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql import types as T


GOLD_DATABASE = os.environ.get("GOLD_DATABASE", "default")
MODEL_DATABASE = os.environ.get("MODEL_DATABASE", GOLD_DATABASE)
GOLD_TABLE_NAME = "gold_beneficiary_year_features"
SELECTION_AUDIT_TABLE = "logreg_variable_selection_audit"
SELECTED_TERMS_TABLE = "logreg_selected_terms"
COLLINEARITY_AUDIT_TABLE = "logreg_candidate_collinearity_audit"
LRT_AUDIT_TABLE = "logreg_nested_lrt_audit"
COEFFICIENT_TABLE = "logreg_selected_coefficients"
MODEL_NAME = "logistic_regression_backward_aic_vif_screened"
RANDOM_SEED = 42
TARGET_QUANTILE = 0.9
SPLIT_STRATEGY = "temporal_target_year_holdout"
SHARED_SPLIT_VERSION = "xxhash64_bene_id_mod_100_v1"
VALIDATION_BUCKET_CUTOFF = 15
MAX_DRIVER_ROWS = int(os.environ.get("MAX_DRIVER_ROWS", "1000000"))
MLFLOW_EXPERIMENT_PATH = os.environ.get(
    "MLFLOW_EXPERIMENT_PATH",
    "/Shared/Project2HighCostClaimClassifier_Experiment",
)


@dataclass(frozen=True)
class TermSpec:
    name: str
    tier: str
    parents: tuple[str, ...] = ()


TERM_SPECS = [
    TermSpec("age_years_imputed", "core"),
    TermSpec("sex", "core"),
    TermSpec("race_code", "core"),
    TermSpec("enrollment_months_count", "core"),
    TermSpec("chronic_condition_count_centered", "core"),
    TermSpec("chronic_condition_count_centered_squared", "functional_form", ("chronic_condition_count_centered",)),
    TermSpec("claims_per_enrollment_month", "utilization"),
    TermSpec("total_claim_days_log1p", "utilization"),
    TermSpec("provider_fragmentation_index", "utilization"),
    TermSpec("any_carrier_claim", "utilization"),
]
TERM_BY_NAME = {term.name: term for term in TERM_SPECS}
CORE_TERMS = {term.name for term in TERM_SPECS if term.tier == "core"}
CANDIDATE_TERMS = [term.name for term in TERM_SPECS]
CANDIDATE_NUMERIC_TERMS = [
    "age_years_imputed",
    "enrollment_months_count",
    "chronic_condition_count_centered",
    "chronic_condition_count_centered_squared",
    "claims_per_enrollment_month",
    "total_claim_days_log1p",
    "provider_fragmentation_index",
    "any_carrier_claim",
]
CANDIDATE_CATEGORICAL_TERMS = ["sex", "race_code"]
REQUIRED_GOLD_COLUMNS = [
    "bene_id",
    "year",
    "annual_claim_cost",
    "chronic_condition_count",
    "total_claim_days",
    "any_carrier_claim",
    "age_years_imputed",
    "enrollment_months_count",
    "claims_per_enrollment_month",
    "provider_fragmentation_index",
    "sex",
    "race_code",
]


def read_gold() -> DataFrame:
    table_name = f"{GOLD_DATABASE}.{GOLD_TABLE_NAME}"
    df = spark.table(table_name)
    missing_columns = sorted(set(REQUIRED_GOLD_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"{table_name} is missing required variable-selection columns: {missing_columns}")
    return df


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
    validation_ids = split_assignments.filter(F.col("shared_split_bucket") < F.lit(VALIDATION_BUCKET_CUTOFF)).select("bene_id")
    return training_pool.join(train_ids, "bene_id", "inner"), training_pool.join(validation_ids, "bene_id", "inner"), test_df


def add_training_target(train_df: DataFrame, validation_df: DataFrame, test_df: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame, float]:
    combined_df = train_df.unionByName(validation_df).unionByName(test_df)
    threshold_df = combined_df.groupBy("target_year").agg(
        F.expr(f"percentile_approx(target_annual_claim_cost, {TARGET_QUANTILE})").alias("target_year_high_cost_threshold")
    )
    threshold_summary = threshold_df.agg(F.avg("target_year_high_cost_threshold").alias("threshold")).collect()[0]

    def with_target(df: DataFrame) -> DataFrame:
        return (
            df.join(threshold_df, "target_year", "left")
            .withColumn(
                "target_cost_within_year_percentile",
                F.percent_rank().over(Window.partitionBy("target_year").orderBy(F.col("target_annual_claim_cost"))),
            )
            .withColumn(
                "label",
                F.when(F.col("target_annual_claim_cost") > F.col("target_year_high_cost_threshold"), 1.0).otherwise(0.0),
            )
        )

    return with_target(train_df), with_target(validation_df), with_target(test_df), float(threshold_summary["threshold"])


def to_pandas_features(df: DataFrame) -> pd.DataFrame:
    selected_df = df.select("bene_id", "year", "target_year", "target_annual_claim_cost", *(REQUIRED_GOLD_COLUMNS[3:] + ["label"]))
    row_count = selected_df.count()
    if row_count > MAX_DRIVER_ROWS:
        raise ValueError(f"Refusing to collect {row_count} rows to pandas; increase MAX_DRIVER_ROWS if intentional.")
    return selected_df.toPandas()


def add_training_derived_terms(
    train_pdf: pd.DataFrame,
    validation_pdf: pd.DataFrame,
    test_pdf: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    chronic_mean = float(train_pdf["chronic_condition_count"].mean())
    for frame in (train_pdf, validation_pdf, test_pdf):
        chronic_centered = frame["chronic_condition_count"] - chronic_mean
        frame["chronic_condition_count_centered"] = chronic_centered
        frame["chronic_condition_count_centered_squared"] = chronic_centered * chronic_centered
        frame["total_claim_days_log1p"] = np.log1p(frame["total_claim_days"].clip(lower=0))
    return train_pdf, validation_pdf, test_pdf, chronic_mean


def term_columns(terms: list[str]) -> tuple[list[str], list[str]]:
    return [term for term in terms if term in CANDIDATE_NUMERIC_TERMS], [term for term in terms if term in CANDIDATE_CATEGORICAL_TERMS]


def build_pipeline(terms: list[str]) -> Pipeline:
    numeric_terms, categorical_terms = term_columns(terms)
    transformers = []
    if numeric_terms:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)), ("scaler", StandardScaler())]),
                numeric_terms,
            )
        )
    if categorical_terms:
        transformers.append(
            (
                "categorical",
                Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="unknown")), ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))]),
                categorical_terms,
            )
        )
    return Pipeline(
        [
            ("preprocessor", ColumnTransformer(transformers=transformers)),
            ("classifier", LogisticRegression(C=1e6, penalty="l2", solver="lbfgs", max_iter=3000, random_state=RANDOM_SEED)),
        ]
    )


def fit_model(train_pdf: pd.DataFrame, terms: list[str]) -> tuple[Pipeline, float, float, float, int]:
    x_train = train_pdf[terms]
    y_train = train_pdf["label"].astype(int)
    model = build_pipeline(terms)
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_train)[:, 1]
    negative_log_likelihood = log_loss(y_train, scores, normalize=False, labels=[0, 1])
    k = len(model.named_steps["preprocessor"].get_feature_names_out()) + 1
    log_likelihood = -float(negative_log_likelihood)
    aic = 2.0 * negative_log_likelihood + 2.0 * k
    bic = 2.0 * negative_log_likelihood + k * math.log(len(y_train))
    return model, float(aic), float(bic), log_likelihood, int(k)


def hierarchy_violations(terms: list[str]) -> list[str]:
    term_set = set(terms)
    return [
        f"{term} missing parent {parent}"
        for term in terms
        for parent in TERM_BY_NAME[term].parents
        if parent not in term_set
    ]


def removable_terms(terms: list[str]) -> list[str]:
    blocked_parents = {parent for term in terms for parent in TERM_BY_NAME[term].parents}
    return sorted(term for term in set(terms) if term not in CORE_TERMS and term not in blocked_parents)


def backward_aic_selection(train_pdf: pd.DataFrame) -> tuple[list[str], list[dict[str, object]], Pipeline]:
    selected_terms = list(CANDIDATE_TERMS)
    violations = hierarchy_violations(selected_terms)
    if violations:
        raise ValueError(f"Initial model violates hierarchy: {violations}")
    best_model, best_aic, best_bic, _best_ll, best_k = fit_model(train_pdf, selected_terms)
    selection_rows = [
        {
            "selection_step": 0,
            "action": "full_vif_screened_model",
            "term_name": ",".join(selected_terms),
            "aic": best_aic,
            "bic": best_bic,
            "parameter_count": best_k,
            "selected_terms": ",".join(selected_terms),
        }
    ]
    step = 1
    while True:
        candidates = removable_terms(selected_terms)
        if not candidates:
            break
        trials = []
        for candidate in candidates:
            trial_terms = [term for term in selected_terms if term != candidate]
            trial_model, trial_aic, trial_bic, _trial_ll, trial_k = fit_model(train_pdf, trial_terms)
            trials.append((candidate, trial_terms, trial_model, trial_aic, trial_bic, trial_k))
        candidate, trial_terms, trial_model, trial_aic, trial_bic, trial_k = min(trials, key=lambda item: item[3])
        if trial_aic >= best_aic:
            break
        selected_terms = trial_terms
        best_model = trial_model
        best_aic = trial_aic
        best_bic = trial_bic
        best_k = trial_k
        selection_rows.append(
            {
                "selection_step": step,
                "action": "remove",
                "term_name": candidate,
                "aic": best_aic,
                "bic": best_bic,
                "parameter_count": best_k,
                "selected_terms": ",".join(selected_terms),
            }
        )
        step += 1
    return selected_terms, selection_rows, best_model


def numeric_vif_rows(train_pdf: pd.DataFrame, terms: list[str], diagnostic_scope: str) -> list[dict[str, object]]:
    numeric_terms = [term for term in terms if term in CANDIDATE_NUMERIC_TERMS]
    rows = []
    if len(numeric_terms) < 2:
        return rows
    corr = train_pdf[numeric_terms].corr().abs()
    for i, left in enumerate(numeric_terms):
        for right in numeric_terms[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and value >= 0.7:
                rows.append(
                    {
                        "diagnostic_scope": diagnostic_scope,
                        "feature_left": left,
                        "feature_right": right,
                        "absolute_correlation": float(value),
                        "vif_value": None,
                        "diagnostic_type": "pairwise_correlation",
                        "collinearity_flag": bool(value >= 0.85),
                    }
                )
    for feature in numeric_terms:
        others = [column for column in numeric_terms if column != feature]
        x = train_pdf[others].fillna(0.0)
        y = train_pdf[feature].fillna(0.0)
        r_squared = float(LinearRegression().fit(x, y).score(x, y))
        vif = math.inf if r_squared >= 0.999999 else 1.0 / (1.0 - r_squared)
        rows.append(
            {
                "diagnostic_scope": diagnostic_scope,
                "feature_left": feature,
                "feature_right": "all_other_numeric_terms_in_scope",
                "absolute_correlation": None,
                "vif_value": float(vif),
                "diagnostic_type": "vif",
                "collinearity_flag": bool(vif >= 10),
            }
        )
    return sorted(rows, key=lambda row: (row["diagnostic_scope"], row["diagnostic_type"], -(row["absolute_correlation"] or row["vif_value"] or 0.0)))


def nested_lrt_rows(train_pdf: pd.DataFrame, selected_terms: list[str]) -> list[dict[str, object]]:
    rows = []
    _full_model, full_aic, _full_bic, full_ll, full_k = fit_model(train_pdf, selected_terms)
    tests = [
        ("chronic_quadratic", ["chronic_condition_count_centered_squared"]),
        ("chronic_block", ["chronic_condition_count_centered", "chronic_condition_count_centered_squared"]),
        (
            "utilization_block",
            [
                "claims_per_enrollment_month",
                "total_claim_days_log1p",
                "provider_fragmentation_index",
                "any_carrier_claim",
            ],
        ),
    ]
    for comparison_name, removed_terms in tests:
        removable = set(removed_terms).issubset(selected_terms)
        if not removable:
            continue
        reduced_terms = [term for term in selected_terms if term not in set(removed_terms)]
        if hierarchy_violations(reduced_terms):
            continue
        _reduced_model, reduced_aic, _reduced_bic, reduced_ll, reduced_k = fit_model(train_pdf, reduced_terms)
        statistic = -2.0 * (reduced_ll - full_ll)
        df = full_k - reduced_k
        rows.append(
            {
                "comparison_name": comparison_name,
                "full_terms": ",".join(selected_terms),
                "reduced_terms": ",".join(reduced_terms),
                "likelihood_ratio_statistic": float(statistic),
                "degrees_of_freedom": int(df),
                "p_value": float(chi2.sf(statistic, df)) if df > 0 else None,
                "full_aic": full_aic,
                "reduced_aic": reduced_aic,
            }
        )
    return rows


def coefficient_rows(model: Pipeline, train_pdf: pd.DataFrame, selected_terms: list[str], run_id: str) -> list[dict[str, object]]:
    x_train = train_pdf[selected_terms]
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    design = preprocessor.transform(x_train)
    if hasattr(design, "toarray"):
        design = design.toarray()
    design_with_intercept = np.column_stack([np.ones(design.shape[0]), design])
    probabilities = np.clip(classifier.predict_proba(design)[:, 1], 1e-6, 1.0 - 1e-6)
    weights = probabilities * (1.0 - probabilities)
    information = design_with_intercept.T @ (design_with_intercept * weights[:, None])
    covariance = np.linalg.pinv(information)
    coefficients = np.concatenate([classifier.intercept_, classifier.coef_[0]])
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    z_values = coefficients / standard_errors
    p_values = 2.0 * norm.sf(np.abs(z_values))
    feature_names = ["intercept", *preprocessor.get_feature_names_out()]
    rows = []
    for name, coefficient, standard_error, z_value, p_value in zip(feature_names, coefficients, standard_errors, z_values, p_values):
        lower = coefficient - 1.96 * standard_error
        upper = coefficient + 1.96 * standard_error
        rows.append(
            {
                "run_id": run_id,
                "coefficient_name": str(name),
                "coefficient": float(coefficient),
                "standard_error": float(standard_error),
                "wald_z": float(z_value),
                "p_value": float(p_value),
                "odds_ratio": float(math.exp(coefficient)),
                "odds_ratio_ci_lower": float(math.exp(lower)),
                "odds_ratio_ci_upper": float(math.exp(upper)),
                "processed_at_utc": None,
            }
        )
    return rows


def choose_decision_threshold(y_true, y_score) -> float:
    candidate_thresholds = np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), np.quantile(y_score, np.linspace(0, 1, 101))]))
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
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    capture_5, lift_5 = top_k_capture_and_lift(y_true, y_score, 0.05)
    capture_10, lift_10 = top_k_capture_and_lift(y_true, y_score, 0.10)
    specificity = true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    return {
        "split_name": split_name,
        "row_count": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "test_error": float(1.0 - accuracy_score(y_true, y_pred)),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "area_under_roc": float(roc_auc_score(y_true, y_score)),
        "area_under_pr": float(average_precision_score(y_true, y_score)),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "top_5_capture_rate": float(capture_5),
        "top_5_lift": float(lift_5),
        "top_10_capture_rate": float(capture_10),
        "top_10_lift": float(lift_10),
    }


def create_selection_audit_df(metrics: list[dict[str, object]], run_id: str, label_threshold: float, decision_threshold: float, selected_terms: list[str]) -> DataFrame:
    schema = T.StructType(
        [
            T.StructField("run_id", T.StringType(), False),
            T.StructField("model_name", T.StringType(), False),
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
            T.StructField("specificity", T.DoubleType(), False),
            T.StructField("area_under_roc", T.DoubleType(), False),
            T.StructField("area_under_pr", T.DoubleType(), False),
            T.StructField("brier_score", T.DoubleType(), False),
            T.StructField("top_5_capture_rate", T.DoubleType(), False),
            T.StructField("top_5_lift", T.DoubleType(), False),
            T.StructField("top_10_capture_rate", T.DoubleType(), False),
            T.StructField("top_10_lift", T.DoubleType(), False),
            T.StructField("high_cost_threshold_train_only", T.DoubleType(), False),
            T.StructField("decision_threshold_from_tuning_split", T.DoubleType(), False),
            T.StructField("selection_method", T.StringType(), False),
            T.StructField("selected_term_count", T.LongType(), False),
            T.StructField("selected_terms", T.StringType(), False),
            T.StructField("shared_split_version", T.StringType(), False),
            T.StructField("processed_at_utc", T.TimestampType(), True),
        ]
    )
    rows = [
        {
            "run_id": run_id,
            "model_name": MODEL_NAME,
            **metric,
            "high_cost_threshold_train_only": label_threshold,
            "decision_threshold_from_tuning_split": decision_threshold,
            "selection_method": "training_only_backward_aic_vif_screened_hierarchy_enforced",
            "selected_term_count": len(selected_terms),
            "selected_terms": ",".join(selected_terms),
            "shared_split_version": SHARED_SPLIT_VERSION,
            "processed_at_utc": None,
        }
        for metric in metrics
    ]
    return spark.createDataFrame(rows, schema=schema).withColumn("processed_at_utc", F.current_timestamp())


def create_selected_terms_df(selection_rows: list[dict[str, object]], run_id: str) -> DataFrame:
    schema = T.StructType(
        [
            T.StructField("run_id", T.StringType(), False),
            T.StructField("selection_step", T.LongType(), False),
            T.StructField("action", T.StringType(), False),
            T.StructField("term_name", T.StringType(), False),
            T.StructField("aic", T.DoubleType(), False),
            T.StructField("bic", T.DoubleType(), False),
            T.StructField("parameter_count", T.LongType(), False),
            T.StructField("selected_terms", T.StringType(), False),
            T.StructField("processed_at_utc", T.TimestampType(), True),
        ]
    )
    return spark.createDataFrame([{"run_id": run_id, **row, "processed_at_utc": None} for row in selection_rows], schema=schema).withColumn("processed_at_utc", F.current_timestamp())


def create_collinearity_df(rows: list[dict[str, object]], run_id: str) -> DataFrame:
    schema = T.StructType(
        [
            T.StructField("run_id", T.StringType(), False),
            T.StructField("diagnostic_scope", T.StringType(), False),
            T.StructField("feature_left", T.StringType(), False),
            T.StructField("feature_right", T.StringType(), False),
            T.StructField("absolute_correlation", T.DoubleType(), True),
            T.StructField("vif_value", T.DoubleType(), True),
            T.StructField("diagnostic_type", T.StringType(), False),
            T.StructField("collinearity_flag", T.BooleanType(), False),
            T.StructField("processed_at_utc", T.TimestampType(), True),
        ]
    )
    output_rows = [{"run_id": run_id, **row, "processed_at_utc": None} for row in rows]
    return spark.createDataFrame(output_rows, schema=schema).withColumn("processed_at_utc", F.current_timestamp())


def create_lrt_df(rows: list[dict[str, object]], run_id: str) -> DataFrame:
    schema = T.StructType(
        [
            T.StructField("run_id", T.StringType(), False),
            T.StructField("comparison_name", T.StringType(), False),
            T.StructField("full_terms", T.StringType(), False),
            T.StructField("reduced_terms", T.StringType(), False),
            T.StructField("likelihood_ratio_statistic", T.DoubleType(), False),
            T.StructField("degrees_of_freedom", T.LongType(), False),
            T.StructField("p_value", T.DoubleType(), True),
            T.StructField("full_aic", T.DoubleType(), False),
            T.StructField("reduced_aic", T.DoubleType(), False),
            T.StructField("processed_at_utc", T.TimestampType(), True),
        ]
    )
    output_rows = [{"run_id": run_id, **row, "processed_at_utc": None} for row in rows] or [
        {
            "run_id": run_id,
            "comparison_name": "none",
            "full_terms": "none",
            "reduced_terms": "none",
            "likelihood_ratio_statistic": 0.0,
            "degrees_of_freedom": 0,
            "p_value": None,
            "full_aic": 0.0,
            "reduced_aic": 0.0,
            "processed_at_utc": None,
        }
    ]
    return spark.createDataFrame(output_rows, schema=schema).withColumn("processed_at_utc", F.current_timestamp())


def create_coefficients_df(rows: list[dict[str, object]]) -> DataFrame:
    schema = T.StructType(
        [
            T.StructField("run_id", T.StringType(), False),
            T.StructField("coefficient_name", T.StringType(), False),
            T.StructField("coefficient", T.DoubleType(), False),
            T.StructField("standard_error", T.DoubleType(), False),
            T.StructField("wald_z", T.DoubleType(), False),
            T.StructField("p_value", T.DoubleType(), False),
            T.StructField("odds_ratio", T.DoubleType(), False),
            T.StructField("odds_ratio_ci_lower", T.DoubleType(), False),
            T.StructField("odds_ratio_ci_upper", T.DoubleType(), False),
            T.StructField("processed_at_utc", T.TimestampType(), True),
        ]
    )
    return spark.createDataFrame(rows, schema=schema).withColumn("processed_at_utc", F.current_timestamp())


def append_table(df: DataFrame, table_name: str) -> None:
    df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(f"{MODEL_DATABASE}.{table_name}")


def main() -> None:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {MODEL_DATABASE}")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)

    modeling_df = build_modeling_frame(read_gold())
    train_df, validation_df, test_df = split_gold_by_time(modeling_df)
    train_df, validation_df, test_df, threshold = add_training_target(train_df, validation_df, test_df)
    train_pdf = to_pandas_features(train_df)
    validation_pdf = to_pandas_features(validation_df)
    test_pdf = to_pandas_features(test_df)
    train_pdf, validation_pdf, test_pdf, chronic_centering_mean = add_training_derived_terms(train_pdf, validation_pdf, test_pdf)

    with mlflow.start_run(run_name="logreg_variable_selection_backward_aic_vif_screened") as run:
        selected_terms, selection_rows, model = backward_aic_selection(train_pdf)
        validation_scores = model.predict_proba(validation_pdf[selected_terms])[:, 1]
        decision_threshold = choose_decision_threshold(validation_pdf["label"].astype(int), validation_scores)
        train_scores = model.predict_proba(train_pdf[selected_terms])[:, 1]
        test_scores = model.predict_proba(test_pdf[selected_terms])[:, 1]
        metrics = [
            evaluate_predictions(train_pdf["label"].astype(int), train_scores, (train_scores >= decision_threshold).astype(int), "train"),
            evaluate_predictions(validation_pdf["label"].astype(int), validation_scores, (validation_scores >= decision_threshold).astype(int), "validation"),
            evaluate_predictions(test_pdf["label"].astype(int), test_scores, (test_scores >= decision_threshold).astype(int), "test"),
        ]
        collinearity_rows = numeric_vif_rows(train_pdf, CANDIDATE_TERMS, "initial_vif_screened_candidate_model")
        collinearity_rows.extend(numeric_vif_rows(train_pdf, selected_terms, "final_selected_model"))
        lrt_rows = nested_lrt_rows(train_pdf, selected_terms)
        coefficients = coefficient_rows(model, train_pdf, selected_terms, run.info.run_id)

        mlflow.log_param("model_family", "logistic_regression")
        mlflow.log_param("statistical_role", "selected_interpretable_baseline")
        mlflow.log_param("selection_method", "training_only_backward_aic_vif_screened_hierarchy_enforced")
        mlflow.log_param("candidate_terms", ",".join(CANDIDATE_TERMS))
        mlflow.log_param("selected_terms", ",".join(selected_terms))
        mlflow.log_param("selected_term_count", len(selected_terms))
        mlflow.log_param("chronic_count_centering", "training_split_mean")
        mlflow.log_param("chronic_count_centering_mean", chronic_centering_mean)
        mlflow.log_param("high_cost_threshold_train_only", threshold)
        mlflow.log_param("decision_threshold_from_tuning_split", decision_threshold)
        mlflow.log_param("threshold_selection_split", "validation")
        mlflow.log_param("split_strategy", SPLIT_STRATEGY)
        mlflow.log_param("shared_split_version", SHARED_SPLIT_VERSION)
        mlflow.log_param("gold_table", f"{GOLD_DATABASE}.{GOLD_TABLE_NAME}")
        mlflow.log_param("final_evaluation_split", "test")
        mlflow.log_param("feature_timing_frame", "current_year_features_predict_next_year_target")
        for metric in metrics:
            prefix = metric["split_name"]
            for name, value in metric.items():
                if name != "split_name":
                    mlflow.log_metric(f"{prefix}_{name}", value)
        signature = infer_signature(train_pdf[selected_terms].head(100), model.predict_proba(train_pdf[selected_terms].head(100)))
        mlflow.sklearn.log_model(model, "model", signature=signature)

        append_table(create_selection_audit_df(metrics, run.info.run_id, threshold, decision_threshold, selected_terms), SELECTION_AUDIT_TABLE)
        append_table(create_selected_terms_df(selection_rows, run.info.run_id), SELECTED_TERMS_TABLE)
        append_table(create_collinearity_df(collinearity_rows, run.info.run_id), COLLINEARITY_AUDIT_TABLE)
        append_table(create_lrt_df(lrt_rows, run.info.run_id), LRT_AUDIT_TABLE)
        append_table(create_coefficients_df(coefficients), COEFFICIENT_TABLE)

        print(f"selected_terms={selected_terms}")
        print(f"selection audit written to {MODEL_DATABASE}.{SELECTION_AUDIT_TABLE}")
        print(f"selected terms written to {MODEL_DATABASE}.{SELECTED_TERMS_TABLE}")
        print(f"collinearity audit written to {MODEL_DATABASE}.{COLLINEARITY_AUDIT_TABLE}")
        print(f"nested LRT audit written to {MODEL_DATABASE}.{LRT_AUDIT_TABLE}")
        print(f"coefficient table written to {MODEL_DATABASE}.{COEFFICIENT_TABLE}")


# COMMAND ----------

main()
