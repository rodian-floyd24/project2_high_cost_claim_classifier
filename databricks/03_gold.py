# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer for CMS Synthetic Claims
# MAGIC
# MAGIC Builds one beneficiary-year feature table from the silver layer for version-1 risk segmentation.

# COMMAND ----------

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from shared.feature_contract import FEATURE_VERSION, CHRONIC_FLAG_FEATURES


SILVER_DATABASE = os.environ.get("SILVER_DATABASE", "default")
GOLD_DATABASE = os.environ.get("GOLD_DATABASE", SILVER_DATABASE)
GOLD_TABLE_NAME = "gold_beneficiary_year_features"
GOLD_FEATURE_VERSION = FEATURE_VERSION
AUDIT_TABLE_NAME = "gold_audit_summary"
AUDIT_STATUS_SUCCEEDED = "succeeded"
AUDIT_STATUS_FAILED_QC = "failed_qc"
AUDIT_STATUS_FAILED_RUNTIME = "failed_runtime"
AUDIT_STATUS_PUBLISHED_AUDIT_FAILED = "published_audit_failed"

SILVER_TABLES = {
    "beneficiaries": "silver_beneficiaries",
    "inpatient_claims": "silver_inpatient_claims",
    "outpatient_claims": "silver_outpatient_claims",
    "carrier_claims": "silver_carrier_claims",
    "pde": "silver_pde",
}

CHRONIC_FLAG_COLUMNS = CHRONIC_FLAG_FEATURES


def read_silver(table_name: str) -> DataFrame:
    return spark.table(f"{SILVER_DATABASE}.{table_name}")


def safe_sum(column_name: str) -> F.Column:
    return F.coalesce(F.sum(F.col(column_name)), F.lit(0.0))


def safe_count_distinct(column_name: str) -> F.Column:
    return F.countDistinct(F.col(column_name))


def safe_rate(numerator: str, denominator: str, scale: float = 1.0) -> F.Column:
    return (
        F.when(
            F.col(denominator).isNull() | (F.col(denominator) == 0),
            F.lit(0.0),
        )
        .otherwise((F.col(numerator).cast("double") / F.col(denominator).cast("double")) * F.lit(scale))
        .cast("double")
    )


def nonnegative_log1p(column_name: str) -> F.Column:
    return F.log1p(F.greatest(F.col(column_name), F.lit(0.0)))


def count_log1p(column_name: str) -> F.Column:
    return F.log1p(F.greatest(F.coalesce(F.col(column_name), F.lit(0)), F.lit(0)))


def nonnegative_value(column_name: str) -> F.Column:
    return F.greatest(F.col(column_name), F.lit(0.0))


def age_band_expr(age_column: str) -> F.Column:
    return (
        F.when(F.col(age_column).isNull(), F.lit("unknown"))
        .when(F.col(age_column) < 65, F.lit("under_65"))
        .when(F.col(age_column) < 75, F.lit("65_74"))
        .when(F.col(age_column) < 85, F.lit("75_84"))
        .otherwise(F.lit("85_plus"))
    )


def age_5yr_band_expr(age_column: str) -> F.Column:
    age = F.col(age_column)
    return (
        F.when(age.isNull(), F.lit("unknown"))
        .when(age < 65, F.lit("under_65"))
        .when(age < 70, F.lit("65_69"))
        .when(age < 75, F.lit("70_74"))
        .when(age < 80, F.lit("75_79"))
        .when(age < 85, F.lit("80_84"))
        .when(age < 90, F.lit("85_89"))
        .otherwise(F.lit("90_plus"))
    )


def chronic_burden_band_expr(count_column: str) -> F.Column:
    count = F.coalesce(F.col(count_column), F.lit(0))
    return (
        F.when(count <= 0, F.lit("0"))
        .when(count <= 2, F.lit("1_2"))
        .when(count <= 5, F.lit("3_5"))
        .otherwise(F.lit("6_plus"))
    )


def enrollment_months_band_expr(months_column: str) -> F.Column:
    months = F.coalesce(F.col(months_column), F.lit(0))
    return (
        F.when(months <= 0, F.lit("0"))
        .when(months <= 3, F.lit("1_3"))
        .when(months <= 6, F.lit("4_6"))
        .when(months <= 11, F.lit("7_11"))
        .otherwise(F.lit("12"))
    )


def build_beneficiary_base(beneficiaries: DataFrame) -> DataFrame:
    age_years = F.floor(
        F.months_between(
            F.make_date(F.col("year"), F.lit(12), F.lit(31)),
            F.col("birth_date"),
        ) / F.lit(12)
    )
    enrollment_months_count = F.greatest(
        F.coalesce(F.col("hi_coverage_months"), F.lit(0)),
        F.coalesce(F.col("smi_coverage_months"), F.lit(0)),
        F.coalesce(F.col("hmo_coverage_months"), F.lit(0)),
        F.coalesce(F.col("plan_coverage_months"), F.lit(0)),
    )

    chronic_flag_exprs = [
        F.coalesce(F.col(column_name).cast("int"), F.lit(0)).alias(column_name)
        for column_name in CHRONIC_FLAG_COLUMNS
    ]

    return (
        beneficiaries.withColumn("age_years", age_years)
        .withColumn("chronic_burden_band", chronic_burden_band_expr("chronic_condition_count"))
        .select(
            "bene_id",
            "year",
            F.col("age_years").cast("int").alias("age_years"),
            age_band_expr("age_years").alias("age_band"),
            age_5yr_band_expr("age_years").alias("age_5yr_band"),
            "sex",
            "race_code",
            "state_code",
            enrollment_months_count.cast("int").alias("enrollment_months_count"),
            "chronic_condition_count",
            "chronic_burden_band",
            *chronic_flag_exprs,
        )
    )


def aggregate_inpatient(df: DataFrame) -> DataFrame:
    return df.groupBy("bene_id", "year").agg(
        safe_count_distinct("claim_id").alias("inpatient_claim_count"),
        safe_sum("payment_amount").alias("inpatient_total_cost"),
        F.coalesce(F.sum(F.col("claim_days")), F.lit(0)).cast("bigint").alias("inpatient_claim_days"),
        F.coalesce(F.sum(F.col("payment_amount")), F.lit(0.0)).alias("inpatient_paid_utilization"),
    )


def aggregate_outpatient(df: DataFrame) -> DataFrame:
    return df.groupBy("bene_id", "year").agg(
        safe_count_distinct("claim_id").alias("outpatient_claim_count"),
        safe_sum("payment_amount").alias("outpatient_total_cost"),
        F.coalesce(F.sum(F.col("claim_days")), F.lit(0)).cast("bigint").alias("outpatient_claim_days"),
        F.coalesce(F.sum(F.col("estimated_line_count")), F.lit(0)).cast("bigint").alias("outpatient_line_count"),
        F.coalesce(
            F.sum(F.when(F.col("emergency_department_claim_flag"), F.lit(1)).otherwise(F.lit(0))),
            F.lit(0),
        )
        .cast("bigint")
        .alias("outpatient_ed_claim_count"),
    )


def aggregate_carrier(df: DataFrame) -> DataFrame:
    return df.groupBy("bene_id", "year").agg(
        safe_count_distinct("claim_id").alias("carrier_claim_count"),
        safe_sum("payment_amount").alias("carrier_total_cost"),
        F.coalesce(F.sum(F.col("estimated_line_count")), F.lit(0)).cast("bigint").alias("carrier_line_count"),
        safe_sum("allowed_amount").alias("carrier_allowed_total"),
    )


def aggregate_pde(df: DataFrame) -> DataFrame:
    return df.groupBy("bene_id", "year").agg(
        safe_count_distinct("pde_id").alias("pde_claim_count"),
        safe_sum("drug_cost").alias("rx_total_cost"),
        F.coalesce(F.sum(F.col("days_supply")), F.lit(0)).cast("bigint").alias("rx_days_supply"),
        safe_sum("patient_pay_amount").alias("rx_patient_pay_total"),
    )


def aggregate_unique_providers(
    inpatient_df: DataFrame,
    outpatient_df: DataFrame,
    carrier_df: DataFrame,
) -> DataFrame:
    provider_events = (
        inpatient_df.select("bene_id", "year", "provider_id")
        .unionByName(outpatient_df.select("bene_id", "year", "provider_id"))
        .unionByName(carrier_df.select("bene_id", "year", "provider_id"))
        .filter(F.col("provider_id").isNotNull())
    )
    return provider_events.groupBy("bene_id", "year").agg(
        safe_count_distinct("provider_id").alias("unique_provider_count")
    )


def build_gold_table() -> DataFrame:
    beneficiaries = read_silver(SILVER_TABLES["beneficiaries"])
    inpatient = read_silver(SILVER_TABLES["inpatient_claims"])
    outpatient = read_silver(SILVER_TABLES["outpatient_claims"])
    carrier = read_silver(SILVER_TABLES["carrier_claims"])
    pde = read_silver(SILVER_TABLES["pde"])

    beneficiary_base = build_beneficiary_base(beneficiaries)

    inpatient_agg = aggregate_inpatient(inpatient)
    outpatient_agg = aggregate_outpatient(outpatient)
    carrier_agg = aggregate_carrier(carrier)
    pde_agg = aggregate_pde(pde)
    provider_agg = aggregate_unique_providers(inpatient, outpatient, carrier)

    gold_df = (
        beneficiary_base.join(inpatient_agg, ["bene_id", "year"], "left")
        .join(outpatient_agg, ["bene_id", "year"], "left")
        .join(carrier_agg, ["bene_id", "year"], "left")
        .join(pde_agg, ["bene_id", "year"], "left")
        .join(provider_agg, ["bene_id", "year"], "left")
        .fillna(
            {
                "inpatient_claim_count": 0,
                "outpatient_claim_count": 0,
                "carrier_claim_count": 0,
                "pde_claim_count": 0,
                "inpatient_total_cost": 0.0,
                "outpatient_total_cost": 0.0,
                "carrier_total_cost": 0.0,
                "rx_total_cost": 0.0,
                "inpatient_claim_days": 0,
                "outpatient_claim_days": 0,
                "outpatient_line_count": 0,
                "outpatient_ed_claim_count": 0,
                "carrier_line_count": 0,
                "carrier_allowed_total": 0.0,
                "rx_days_supply": 0,
                "rx_patient_pay_total": 0.0,
                "unique_provider_count": 0,
            }
        )
        .withColumn(
            "total_claim_days",
            F.col("inpatient_claim_days") + F.col("outpatient_claim_days"),
        )
        .drop("inpatient_claim_days", "outpatient_claim_days")
        .withColumn(
            "annual_claim_cost",
            F.col("inpatient_total_cost")
            + F.col("outpatient_total_cost")
            + F.col("carrier_total_cost")
            + F.col("rx_total_cost"),
        )
        .withColumn(
            "total_claim_count",
            F.col("inpatient_claim_count")
            + F.col("outpatient_claim_count")
            + F.col("carrier_claim_count")
            + F.col("pde_claim_count"),
        )
        .withColumn("cost_per_enrollment_month", safe_rate("annual_claim_cost", "enrollment_months_count"))
        .withColumn("claims_per_enrollment_month", safe_rate("total_claim_count", "enrollment_months_count"))
        .withColumn("claim_days_per_enrollment_month", safe_rate("total_claim_days", "enrollment_months_count"))
        .withColumn("providers_per_enrollment_month", safe_rate("unique_provider_count", "enrollment_months_count"))
        .withColumn("provider_fragmentation_index", safe_rate("unique_provider_count", "total_claim_count"))
        .withColumn("full_year_enrolled_flag", (F.col("enrollment_months_count") == 12).cast("int"))
        .withColumn(
            "partial_year_enrolled_flag",
            ((F.col("enrollment_months_count") > 0) & (F.col("enrollment_months_count") < 12)).cast("int"),
        )
        .withColumn("zero_enrollment_flag", (F.col("enrollment_months_count") <= 0).cast("int"))
        .withColumn(
            "low_enrollment_flag",
            ((F.col("enrollment_months_count") > 0) & (F.col("enrollment_months_count") <= 3)).cast("int"),
        )
        .withColumn("enrollment_months_band", enrollment_months_band_expr("enrollment_months_count"))
        .withColumn("enrollment_fraction", F.col("enrollment_months_count").cast("double") / F.lit(12.0))
        .withColumn("annualized_cost_per_enrolled_month", safe_rate("annual_claim_cost", "enrollment_months_count", 12.0))
        .withColumn("annualized_claims_per_enrolled_month", safe_rate("total_claim_count", "enrollment_months_count", 12.0))
        .withColumn("inpatient_claims_per_enrollment_month", safe_rate("inpatient_claim_count", "enrollment_months_count"))
        .withColumn("outpatient_claims_per_enrollment_month", safe_rate("outpatient_claim_count", "enrollment_months_count"))
        .withColumn("carrier_claims_per_enrollment_month", safe_rate("carrier_claim_count", "enrollment_months_count"))
        .withColumn("rx_fills_per_enrollment_month", safe_rate("pde_claim_count", "enrollment_months_count"))
        .withColumn("outpatient_ed_claims_per_enrollment_month", safe_rate("outpatient_ed_claim_count", "enrollment_months_count"))
        .withColumn("rx_days_supply_per_enrollment_month", safe_rate("rx_days_supply", "enrollment_months_count"))
        .withColumn("avg_inpatient_cost_per_claim", safe_rate("inpatient_total_cost", "inpatient_claim_count"))
        .withColumn("avg_outpatient_cost_per_claim", safe_rate("outpatient_total_cost", "outpatient_claim_count"))
        .withColumn("avg_carrier_cost_per_claim", safe_rate("carrier_total_cost", "carrier_claim_count"))
        .withColumn("avg_rx_cost_per_fill", safe_rate("rx_total_cost", "pde_claim_count"))
        .withColumn("outpatient_lines_per_claim", safe_rate("outpatient_line_count", "outpatient_claim_count"))
        .withColumn("carrier_lines_per_claim", safe_rate("carrier_line_count", "carrier_claim_count"))
        .withColumn("any_inpatient_claim", (F.col("inpatient_claim_count") > 0).cast("int"))
        .withColumn("any_outpatient_claim", (F.col("outpatient_claim_count") > 0).cast("int"))
        .withColumn("any_carrier_claim", (F.col("carrier_claim_count") > 0).cast("int"))
        .withColumn("any_pde_claim", (F.col("pde_claim_count") > 0).cast("int"))
        .withColumn("any_outpatient_ed_claim", (F.col("outpatient_ed_claim_count") > 0).cast("int"))
        .withColumn("multiple_provider_flag", (F.col("unique_provider_count") > 1).cast("int"))
        .withColumn(
            "multi_setting_utilization_flag",
            (
                F.col("any_inpatient_claim")
                + F.col("any_outpatient_claim")
                + F.col("any_carrier_claim")
                + F.col("any_pde_claim")
                >= F.lit(2)
            ).cast("int"),
        )
        .withColumn("inpatient_cost_log1p", nonnegative_log1p("inpatient_total_cost"))
        .withColumn("outpatient_cost_log1p", nonnegative_log1p("outpatient_total_cost"))
        .withColumn("carrier_cost_log1p", nonnegative_log1p("carrier_total_cost"))
        .withColumn("rx_cost_log1p", nonnegative_log1p("rx_total_cost"))
        .withColumn("annual_cost_log1p", nonnegative_log1p("annual_claim_cost"))
        .withColumn("inpatient_claim_count_log1p", count_log1p("inpatient_claim_count"))
        .withColumn("outpatient_claim_count_log1p", count_log1p("outpatient_claim_count"))
        .withColumn("carrier_claim_count_log1p", count_log1p("carrier_claim_count"))
        .withColumn("pde_claim_count_log1p", count_log1p("pde_claim_count"))
        .withColumn("total_claim_count_log1p", count_log1p("total_claim_count"))
        .withColumn("unique_provider_count_log1p", count_log1p("unique_provider_count"))
        .withColumn(
            "annual_cost_year_percentile",
            F.percent_rank().over(Window.partitionBy("year").orderBy(F.col("annual_claim_cost"))),
        )
        .withColumn(
            "annual_cost_year_decile",
            F.ntile(10).over(Window.partitionBy("year").orderBy(F.col("annual_claim_cost"))),
        )
        .withColumn(
            "annual_cost_year_median",
            F.expr("percentile_approx(annual_claim_cost, 0.5)").over(Window.partitionBy("year")),
        )
        .withColumn(
            "annual_cost_to_year_median",
            F.when(
                F.col("annual_cost_year_median").isNull() | (F.col("annual_cost_year_median") == 0),
                F.lit(0.0),
            ).otherwise(F.col("annual_claim_cost") / F.col("annual_cost_year_median")),
        )
        .withColumn("inpatient_positive_cost", nonnegative_value("inpatient_total_cost"))
        .withColumn("outpatient_positive_cost", nonnegative_value("outpatient_total_cost"))
        .withColumn("carrier_positive_cost", nonnegative_value("carrier_total_cost"))
        .withColumn("rx_positive_cost", nonnegative_value("rx_total_cost"))
        .withColumn(
            "positive_cost_total",
            F.col("inpatient_positive_cost")
            + F.col("outpatient_positive_cost")
            + F.col("carrier_positive_cost")
            + F.col("rx_positive_cost"),
        )
        .withColumn("inpatient_cost_share", safe_rate("inpatient_positive_cost", "positive_cost_total"))
        .withColumn("outpatient_cost_share", safe_rate("outpatient_positive_cost", "positive_cost_total"))
        .withColumn("carrier_cost_share", safe_rate("carrier_positive_cost", "positive_cost_total"))
        .withColumn("rx_cost_share", safe_rate("rx_positive_cost", "positive_cost_total"))
        .withColumn(
            "chronic_burden_age_band",
            F.concat_ws("__", F.col("chronic_burden_band"), F.col("age_band")),
        )
        .withColumn(
            "chronic_burden_age_5yr_band",
            F.concat_ws("__", F.col("chronic_burden_band"), F.col("age_5yr_band")),
        )
        .withColumn(
            "sex_chronic_burden_band",
            F.concat_ws("__", F.col("sex"), F.col("chronic_burden_band")),
        )
        .withColumn("age_years_imputed", F.coalesce(F.col("age_years"), F.lit(75)))
        .withColumn("age_over_65", F.greatest(F.col("age_years_imputed") - F.lit(65), F.lit(0)))
        .withColumn("age_over_75", F.greatest(F.col("age_years_imputed") - F.lit(75), F.lit(0)))
        .withColumn("age_over_85", F.greatest(F.col("age_years_imputed") - F.lit(85), F.lit(0)))
        .withColumn("age_squared", F.col("age_years_imputed") * F.col("age_years_imputed"))
        .withColumn("chronic_condition_count_squared", F.col("chronic_condition_count") * F.col("chronic_condition_count"))
        .withColumn(
            "claims_per_month_chronic_count_interaction",
            F.col("claims_per_enrollment_month") * F.col("chronic_condition_count"),
        )
        .withColumn(
            "providers_per_month_chronic_count_interaction",
            F.col("providers_per_enrollment_month") * F.col("chronic_condition_count"),
        )
        .withColumn("age_inpatient_claim_interaction", F.col("age_years_imputed") * F.col("inpatient_claim_count"))
        .withColumn("age_total_claim_interaction", F.col("age_years_imputed") * F.col("total_claim_count"))
        .withColumn("age_chronic_count_interaction", F.col("age_years_imputed") * F.col("chronic_condition_count"))
        .withColumn(
            "sex_male_chronic_count_interaction",
            F.when(F.col("sex") == "male", F.col("chronic_condition_count")).otherwise(F.lit(0)),
        )
        .withColumn(
            "sex_female_chronic_count_interaction",
            F.when(F.col("sex") == "female", F.col("chronic_condition_count")).otherwise(F.lit(0)),
        )
        .withColumn(
            "enrollment_months_total_claim_interaction",
            F.col("enrollment_months_count") * F.col("total_claim_count"),
        )
        .withColumn(
            "enrollment_months_inpatient_interaction",
            F.col("enrollment_months_count") * F.col("inpatient_claim_count"),
        )
        .withColumn(
            "chronic_count_age_under_65",
            F.when(F.col("age_band") == "under_65", F.col("chronic_condition_count")).otherwise(F.lit(0)),
        )
        .withColumn(
            "chronic_count_age_65_74",
            F.when(F.col("age_band") == "65_74", F.col("chronic_condition_count")).otherwise(F.lit(0)),
        )
        .withColumn(
            "chronic_count_age_75_84",
            F.when(F.col("age_band") == "75_84", F.col("chronic_condition_count")).otherwise(F.lit(0)),
        )
        .withColumn(
            "chronic_count_age_85_plus",
            F.when(F.col("age_band") == "85_plus", F.col("chronic_condition_count")).otherwise(F.lit(0)),
        )
    )

    beneficiary_window = Window.partitionBy("bene_id").orderBy("year")
    current_year_high_cost = (F.col("annual_cost_year_decile") == F.lit(10)).cast("int")
    prior_year_high_cost = F.coalesce(F.lag(current_year_high_cost, 1).over(beneficiary_window), F.lit(0))
    two_year_high_cost_count = current_year_high_cost + prior_year_high_cost
    gold_df = (
        gold_df.withColumn("has_prior_year", F.lag("year", 1).over(beneficiary_window).isNotNull().cast("int"))
        .withColumn(
            "prior_year_annual_claim_cost",
            F.coalesce(F.lag("annual_claim_cost", 1).over(beneficiary_window), F.lit(0.0)),
        )
        .withColumn(
            "prior_year_inpatient_claim_count",
            F.coalesce(F.lag("inpatient_claim_count", 1).over(beneficiary_window), F.lit(0)),
        )
        .withColumn(
            "prior_year_total_claim_count",
            F.coalesce(F.lag("total_claim_count", 1).over(beneficiary_window), F.lit(0)),
        )
        .withColumn(
            "prior_year_enrollment_months_count",
            F.coalesce(F.lag("enrollment_months_count", 1).over(beneficiary_window), F.lit(0)),
        )
        .withColumn("current_year_high_cost_indicator", current_year_high_cost)
        .withColumn("prior_year_high_cost_indicator", prior_year_high_cost)
        .withColumn(
            "two_year_avg_annual_claim_cost",
            (
                F.col("annual_claim_cost")
                + F.when(F.col("has_prior_year") == 1, F.col("prior_year_annual_claim_cost")).otherwise(F.lit(0.0))
            )
            / (F.col("has_prior_year") + F.lit(1)),
        )
        .withColumn("cost_trend_difference", F.col("annual_claim_cost") - F.col("prior_year_annual_claim_cost"))
        .withColumn("cost_trend_ratio", safe_rate("annual_claim_cost", "prior_year_annual_claim_cost"))
        .withColumn(
            "inpatient_claim_count_change",
            F.col("inpatient_claim_count") - F.col("prior_year_inpatient_claim_count"),
        )
        .withColumn("total_claim_count_change", F.col("total_claim_count") - F.col("prior_year_total_claim_count"))
        .withColumn(
            "utilization_trend",
            F.when(F.col("has_prior_year") == 0, F.lit("no_prior"))
            .when(F.col("total_claim_count_change") > 0, F.lit("rising"))
            .when(F.col("total_claim_count_change") < 0, F.lit("falling"))
            .otherwise(F.lit("flat")),
        )
        .withColumn("high_cost_last_2yr_count", two_year_high_cost_count)
        .withColumn("high_cost_1_of_last_2yr", (two_year_high_cost_count >= 1).cast("int"))
        .withColumn("high_cost_2_of_last_2yr", (two_year_high_cost_count >= 2).cast("int"))
    )

    gold_df = gold_df.select(
        "bene_id",
        "year",
        "annual_claim_cost",
        "age_years",
        "age_band",
        "age_5yr_band",
        "sex",
        "race_code",
        "state_code",
        "enrollment_months_count",
        "full_year_enrolled_flag",
        "partial_year_enrolled_flag",
        "zero_enrollment_flag",
        "low_enrollment_flag",
        "enrollment_months_band",
        "enrollment_fraction",
        "annualized_cost_per_enrolled_month",
        "annualized_claims_per_enrolled_month",
        "chronic_condition_count",
        "chronic_burden_band",
        *CHRONIC_FLAG_COLUMNS,
        "chronic_burden_age_band",
        "chronic_burden_age_5yr_band",
        "sex_chronic_burden_band",
        "age_years_imputed",
        "age_over_65",
        "age_over_75",
        "age_over_85",
        "age_squared",
        "chronic_condition_count_squared",
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
        "inpatient_cost_log1p",
        "outpatient_cost_log1p",
        "carrier_cost_log1p",
        "rx_cost_log1p",
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
        "utilization_trend",
        "high_cost_last_2yr_count",
        "high_cost_1_of_last_2yr",
        "high_cost_2_of_last_2yr",
        "inpatient_cost_share",
        "outpatient_cost_share",
        "carrier_cost_share",
        "rx_cost_share",
    )

    return gold_df


def write_gold_table(df: DataFrame) -> None:
    (
        df.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{GOLD_DATABASE}.{GOLD_TABLE_NAME}")
    )
    spark.sql(
        f"""
        ALTER TABLE {GOLD_DATABASE}.{GOLD_TABLE_NAME}
        SET TBLPROPERTIES (
            'gold_feature_version' = '{GOLD_FEATURE_VERSION}',
            'grain' = 'beneficiary_year',
            'primary_key' = 'bene_id,year',
            'source_system' = 'cms_desynpuf',
            'gold_feature_source_notebook' = 'databricks/03_gold.py'
        )
        """
    )


def append_audit_table(df: DataFrame) -> None:
    (
        df.write.format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(f"{GOLD_DATABASE}.{AUDIT_TABLE_NAME}")
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def compute_quality_metrics(gold_df: DataFrame) -> dict[str, int]:
    metrics = gold_df.agg(
        F.count("*").alias("row_count"),
        F.countDistinct("bene_id").alias("distinct_bene_id_count"),
        F.coalesce(
            F.sum(F.when(F.col("bene_id").isNotNull() & F.col("year").isNotNull(), F.lit(1)).otherwise(F.lit(0))),
            F.lit(0),
        ).alias("nonnull_bene_year_row_count"),
        F.countDistinct("bene_id", "year").alias("distinct_bene_year_count"),
        F.coalesce(
            F.sum(F.when(F.col("bene_id").isNull() | F.col("year").isNull(), F.lit(1)).otherwise(F.lit(0))),
            F.lit(0),
        ).alias("missing_bene_year_key_count"),
        F.coalesce(
            F.sum(F.when(F.col("enrollment_months_count") <= 0, F.lit(1)).otherwise(F.lit(0))),
            F.lit(0),
        ).alias("nonpositive_enrollment_month_count"),
        F.coalesce(
            F.sum(
                F.when(
                    (F.col("enrollment_months_count") < 0) | (F.col("enrollment_months_count") > 12),
                    F.lit(1),
                ).otherwise(F.lit(0))
            ),
            F.lit(0),
        ).alias("invalid_enrollment_month_count"),
        F.coalesce(
            F.sum(
                F.when(
                    (F.col("chronic_condition_count") < 0) | (F.col("chronic_condition_count") > 11),
                    F.lit(1),
                ).otherwise(F.lit(0))
            ),
            F.lit(0),
        ).alias("invalid_chronic_condition_count"),
        F.coalesce(
            F.sum(
                F.when(
                    (F.col("inpatient_total_cost") < 0)
                    | (F.col("outpatient_total_cost") < 0)
                    | (F.col("carrier_total_cost") < 0)
                    | (F.col("rx_total_cost") < 0),
                    F.lit(1),
                ).otherwise(F.lit(0))
            ),
            F.lit(0),
        ).alias("negative_component_cost_count"),
    ).collect()[0]

    nonnull_bene_year_row_count = int(metrics["nonnull_bene_year_row_count"])
    distinct_bene_year_count = int(metrics["distinct_bene_year_count"])

    return {
        "row_count": int(metrics["row_count"]),
        "distinct_bene_id_count": int(metrics["distinct_bene_id_count"]),
        "distinct_bene_year_count": distinct_bene_year_count,
        "duplicate_bene_year_count": nonnull_bene_year_row_count - distinct_bene_year_count,
        "missing_bene_year_key_count": int(metrics["missing_bene_year_key_count"]),
        "nonpositive_enrollment_month_count": int(metrics["nonpositive_enrollment_month_count"]),
        "invalid_enrollment_month_count": int(metrics["invalid_enrollment_month_count"]),
        "invalid_chronic_condition_count": int(metrics["invalid_chronic_condition_count"]),
        "negative_component_cost_count": int(metrics["negative_component_cost_count"]),
    }


def publish_blocking_failure(metrics: dict[str, int]) -> str | None:
    if metrics["missing_bene_year_key_count"] > 0:
        return f"found {metrics['missing_bene_year_key_count']} rows with missing bene_id/year keys"
    if metrics["duplicate_bene_year_count"] > 0:
        return f"found {metrics['duplicate_bene_year_count']} duplicate bene_id/year rows"
    if metrics["row_count"] < 1:
        return "gold table has no rows"
    if metrics["invalid_enrollment_month_count"] > 0:
        return f"found {metrics['invalid_enrollment_month_count']} rows with enrollment months outside 0..12"
    if metrics["invalid_chronic_condition_count"] > 0:
        return f"found {metrics['invalid_chronic_condition_count']} rows with chronic condition count outside 0..11"
    if metrics["negative_component_cost_count"] > 0:
        return f"found {metrics['negative_component_cost_count']} rows with negative component costs"
    return None


def empty_quality_metrics() -> dict[str, int | None]:
    return {
        "row_count": None,
        "distinct_bene_id_count": None,
        "distinct_bene_year_count": None,
        "duplicate_bene_year_count": None,
        "missing_bene_year_key_count": None,
        "nonpositive_enrollment_month_count": None,
        "invalid_enrollment_month_count": None,
        "invalid_chronic_condition_count": None,
        "negative_component_cost_count": None,
    }


def build_audit_df(
    metrics: dict[str, int | None],
    run_id: str,
    status: str,
    failure_reason: str | None = None,
) -> DataFrame:
    schema = T.StructType(
        [
            T.StructField("run_id", T.StringType(), False),
            T.StructField("status", T.StringType(), False),
            T.StructField("failure_reason", T.StringType(), True),
            T.StructField("gold_table", T.StringType(), False),
            T.StructField("source_database", T.StringType(), False),
            T.StructField("source_tables", T.StringType(), False),
            T.StructField("row_count", T.LongType(), True),
            T.StructField("distinct_bene_id_count", T.LongType(), True),
            T.StructField("distinct_bene_year_count", T.LongType(), True),
            T.StructField("duplicate_bene_year_count", T.LongType(), True),
            T.StructField("missing_bene_year_key_count", T.LongType(), True),
            T.StructField("nonpositive_enrollment_month_count", T.LongType(), True),
            T.StructField("invalid_enrollment_month_count", T.LongType(), True),
            T.StructField("invalid_chronic_condition_count", T.LongType(), True),
            T.StructField("negative_component_cost_count", T.LongType(), True),
            T.StructField("processed_at_utc", T.TimestampType(), True),
        ]
    )

    return spark.createDataFrame(
        [
            {
                "run_id": run_id,
                "status": status,
                "failure_reason": failure_reason,
                "gold_table": f"{GOLD_DATABASE}.{GOLD_TABLE_NAME}",
                "source_database": SILVER_DATABASE,
                "source_tables": ",".join(
                    f"{SILVER_DATABASE}.{table_name}" for table_name in sorted(SILVER_TABLES.values())
                ),
                "row_count": metrics["row_count"],
                "distinct_bene_id_count": metrics["distinct_bene_id_count"],
                "distinct_bene_year_count": metrics["distinct_bene_year_count"],
                "duplicate_bene_year_count": metrics["duplicate_bene_year_count"],
                "missing_bene_year_key_count": metrics["missing_bene_year_key_count"],
                "nonpositive_enrollment_month_count": metrics["nonpositive_enrollment_month_count"],
                "invalid_enrollment_month_count": metrics["invalid_enrollment_month_count"],
                "invalid_chronic_condition_count": metrics["invalid_chronic_condition_count"],
                "negative_component_cost_count": metrics["negative_component_cost_count"],
                "processed_at_utc": utc_now(),
            }
        ],
        schema=schema,
    )


def main() -> None:
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {GOLD_DATABASE}")
    run_id = str(uuid.uuid4())
    metrics = empty_quality_metrics()
    failure_audit_written = False
    gold_published = False

    try:
        gold_df = build_gold_table()
        metrics = compute_quality_metrics(gold_df)
        failure_reason = publish_blocking_failure(metrics)
        if failure_reason:
            audit_df = build_audit_df(metrics, run_id, AUDIT_STATUS_FAILED_QC, failure_reason)
            append_audit_table(audit_df)
            failure_audit_written = True
            raise ValueError(f"Refusing to publish {GOLD_DATABASE}.{GOLD_TABLE_NAME}: {failure_reason}")

        write_gold_table(gold_df)
        gold_published = True
        audit_df = build_audit_df(metrics, run_id, AUDIT_STATUS_SUCCEEDED)
        append_audit_table(audit_df)
    except Exception as exc:
        if not failure_audit_written:
            failure_status = AUDIT_STATUS_PUBLISHED_AUDIT_FAILED if gold_published else AUDIT_STATUS_FAILED_RUNTIME
            audit_df = build_audit_df(metrics, run_id, failure_status, str(exc))
            try:
                append_audit_table(audit_df)
            except Exception as audit_exc:
                print(
                    f"failed to append audit row for run_id={run_id} "
                    f"status={failure_status}: {audit_exc}"
                )
        raise

    print(
        f"run_id={run_id} status={AUDIT_STATUS_SUCCEEDED} gold table={GOLD_DATABASE}.{GOLD_TABLE_NAME} "
        f"row_count={metrics['row_count']} "
        f"distinct_bene_id_count={metrics['distinct_bene_id_count']} "
        f"distinct_bene_year_count={metrics['distinct_bene_year_count']} "
        f"duplicate_bene_year_count={metrics['duplicate_bene_year_count']} "
        f"missing_bene_year_key_count={metrics['missing_bene_year_key_count']} "
        f"nonpositive_enrollment_month_count={metrics['nonpositive_enrollment_month_count']} "
        f"invalid_enrollment_month_count={metrics['invalid_enrollment_month_count']} "
        f"invalid_chronic_condition_count={metrics['invalid_chronic_condition_count']} "
        f"negative_component_cost_count={metrics['negative_component_cost_count']}"
    )
    print(f"gold audit summary appended to {GOLD_DATABASE}.{AUDIT_TABLE_NAME}")


# COMMAND ----------

main()
