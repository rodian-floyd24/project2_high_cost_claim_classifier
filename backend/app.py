from __future__ import annotations

import json
import math
import sys
import types
from functools import lru_cache
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, model_validator
from shared.feature_contract import (
    FEATURE_CONTRACT_VERSION,
    SERVED_MODEL_FEATURE_ORDER,
    MODEL_INT32_FIELDS,
    MODEL_INT64_FIELDS,
)

from .explanations import reason_code_version, top_risk_drivers
from .rl.config import SIMULATION_DISCLAIMER
from .rl.mdp import assign_risk_tier
from .rl.policy import action_values_for_profile, compare_all_actions, profile_to_state
from .rl.q_learning import load_q_table
from .schemas import input_review_flags, validate_profile_values
from .scoring import (
    MODEL_NAME,
    MODEL_VERSION,
    deterministic_feature_order,
    load_model_metadata,
    compute_risk_score,
    assign_risk_tier,
    generate_reason_codes,
    load_reference_distribution,
)


MODEL_PATH = Path(__file__).resolve().parent / "model_artifacts" / "model"
Q_TABLE_PATH = Path(__file__).resolve().parent / "model_artifacts" / "q_table.json"
RL_METADATA_PATH = Path(__file__).resolve().parent / "model_artifacts" / "rl_metadata.json"
MODEL_METRICS_PATH = Path(__file__).resolve().parent / "model_artifacts" / "model_metrics.json"
MODEL_DECISION_THRESHOLD = 0.20


class BeneficiaryProfile(BaseModel):
    age_band: str = Field(pattern="^(under_65|65_74|75_84|85_plus|unknown)$")
    age_years: int | None = Field(default=None, ge=0, le=115)
    sex: str = Field(pattern="^(male|female|unknown)$")
    race_code: str = Field(min_length=1, max_length=8)
    state_code: str = Field(min_length=1, max_length=8)
    enrollment_months_count: int = Field(ge=0, le=12)
    chronic_condition_count: int = Field(ge=0, le=11)
    alzheimers_flag: bool = False
    chf_flag: bool = False
    chronic_kidney_disease_flag: bool = False
    cancer_flag: bool = False
    copd_flag: bool = False
    depression_flag: bool = False
    diabetes_flag: bool = False
    ischemic_heart_disease_flag: bool = False
    osteoporosis_flag: bool = False
    rheumatoid_arthritis_oa_flag: bool = False
    stroke_tia_flag: bool = False
    inpatient_claim_count: int = Field(ge=0)
    outpatient_claim_count: int = Field(ge=0)
    carrier_claim_count: int = Field(ge=0)
    pde_claim_count: int = Field(ge=0)
    outpatient_ed_claim_count: int = Field(default=0, ge=0)
    outpatient_line_count: int = Field(default=0, ge=0)
    carrier_line_count: int = Field(default=0, ge=0)
    rx_days_supply: int = Field(default=0, ge=0)
    total_claim_days: int = Field(ge=0)
    unique_provider_count: int = Field(ge=0)
    rx_total_cost: float = Field(ge=0.0)
    inpatient_total_cost: float
    outpatient_total_cost: float
    carrier_total_cost: float
    carrier_allowed_total: float = Field(default=0.0, ge=0.0)
    rx_patient_pay_total: float = Field(default=0.0, ge=0.0)
    annual_cost_year_percentile: float = Field(default=0.5, ge=0.0, le=1.0)
    annual_cost_year_decile: int = Field(default=5, ge=1, le=10)
    annual_cost_to_year_median: float = Field(default=1.0, ge=0.0)
    has_prior_year: int = Field(default=0, ge=0, le=1)
    prior_year_annual_claim_cost: float = Field(default=0.0, ge=0.0)
    prior_year_inpatient_claim_count: int = Field(default=0, ge=0)
    prior_year_total_claim_count: int = Field(default=0, ge=0)
    prior_year_enrollment_months_count: int = Field(default=0, ge=0, le=12)
    current_year_high_cost_indicator: int = Field(default=0, ge=0, le=1)
    prior_year_high_cost_indicator: int = Field(default=0, ge=0, le=1)
    prior_intervention_status: str = Field(
        default="none",
        pattern="^(none|recent_low_touch|recent_intensive)$",
    )

    @model_validator(mode="after")
    def validate_actuarial_bounds(self):
        validate_profile_values(self.model_dump())
        return self


class PredictionMetrics(BaseModel):
    raw_model_probability: float
    calibrated_probability: float
    risk_score_0_100: int
    risk_tier: str
    intervention_flag: bool
    decision_threshold: float


class PredictionMetadata(BaseModel):
    model_name: str
    feature_contract_version: str
    calibration_method: str
    threshold_source: str
    split_version: str
    score_transform: str


class PredictionResponse(BaseModel):
    model_name: str
    model_version: str
    raw_model_probability: float
    calibrated_probability: float
    risk_score_0_100: int
    risk_tier: str
    predicted_high_cost: bool
    intervention_flag: bool
    decision_threshold: float
    threshold_source: str
    reason_codes: list[str]
    feature_contract_version: str
    calibration_method: str
    split_version: str
    score_transform: str
    reference_distribution_available: bool
    risk_probability: float
    risk_score: float
    prediction: PredictionMetrics
    metadata: PredictionMetadata
    annual_claim_cost_proxy: float
    cost_mix: dict[str, float]
    engineered_features: dict[str, float | int | str]


class StateResponse(BaseModel):
    risk_probability: float
    current_state: dict[str, object]
    disclaimer: str


class ActionValueResponse(BaseModel):
    action: str
    action_label: str
    q_value: float
    expected_next_risk_probability: float
    expected_immediate_reward: float


class RecommendationResponse(BaseModel):
    risk_probability: float
    risk_tier: str
    current_state: dict[str, object]
    recommended_action: str
    recommended_action_label: str
    recommended_action_display: str
    expected_long_run_value: float
    q_values: dict[str, float]
    action_values: list[ActionValueResponse]
    policy_explanation: str
    disclaimer: str


class SimulationComparisonResponse(BaseModel):
    action: str
    action_label: str
    q_value: float
    expected_next_risk_probability: float
    expected_immediate_reward: float
    expected_risk_delta: float
    is_recommended: bool


class SimulationResponse(BaseModel):
    baseline_risk_probability: float
    risk_tier: str
    current_state: dict[str, object]
    recommended_action: str
    recommended_action_label: str
    recommended_action_display: str
    q_values: dict[str, float]
    comparisons: list[SimulationComparisonResponse]
    policy_explanation: str
    disclaimer: str


class DecisionSupportResponse(BaseModel):
    prediction: PredictionResponse
    state: StateResponse
    recommendation: RecommendationResponse
    simulation: SimulationResponse
    disclaimer: str


def safe_rate(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else float(numerator) / float(denominator)


def nonnegative_log1p(value: float) -> float:
    return math.log1p(max(float(value), 0.0))


def chronic_burden_band(chronic_condition_count: int) -> str:
    if chronic_condition_count <= 0:
        return "0"
    if chronic_condition_count <= 2:
        return "1_2"
    if chronic_condition_count <= 5:
        return "3_5"
    return "6_plus"


def impute_age_years(age_band: str, age_years: int | None) -> int:
    if age_years is not None:
        return int(age_years)
    return {
        "under_65": 60,
        "65_74": 70,
        "75_84": 80,
        "85_plus": 88,
        "unknown": 75,
    }[age_band]


def age_5yr_band(age_years: int | None, age_band: str) -> str:
    if age_years is None:
        return "unknown" if age_band == "unknown" else age_band
    if age_years < 65:
        return "under_65"
    if age_years < 70:
        return "65_69"
    if age_years < 75:
        return "70_74"
    if age_years < 80:
        return "75_79"
    if age_years < 85:
        return "80_84"
    if age_years < 90:
        return "85_89"
    return "90_plus"


def enrollment_months_band(enrollment_months: int) -> str:
    if enrollment_months <= 0:
        return "0"
    if enrollment_months <= 3:
        return "1_3"
    if enrollment_months <= 6:
        return "4_6"
    if enrollment_months <= 11:
        return "7_11"
    return "12"


def utilization_trend(total_claim_count: int, prior_year_total_claim_count: int, has_prior_year: int) -> str:
    if not has_prior_year:
        return "no_prior"
    if total_claim_count > prior_year_total_claim_count:
        return "rising"
    if total_claim_count < prior_year_total_claim_count:
        return "falling"
    return "flat"


def build_feature_row(profile: BeneficiaryProfile) -> dict[str, float | int | str]:
    annual_claim_cost = (
        float(profile.rx_total_cost)
        + float(profile.inpatient_total_cost)
        + float(profile.outpatient_total_cost)
        + float(profile.carrier_total_cost)
    )
    total_claim_count = (
        int(profile.inpatient_claim_count)
        + int(profile.outpatient_claim_count)
        + int(profile.carrier_claim_count)
        + int(profile.pde_claim_count)
    )
    enrollment_months = int(profile.enrollment_months_count)
    positive_cost_total = (
        max(float(profile.inpatient_total_cost), 0.0)
        + max(float(profile.outpatient_total_cost), 0.0)
        + max(float(profile.carrier_total_cost), 0.0)
        + max(float(profile.rx_total_cost), 0.0)
    )
    chronic_count = int(profile.chronic_condition_count)
    burden_band = chronic_burden_band(chronic_count)
    age_years = impute_age_years(profile.age_band, profile.age_years)
    age_missing_flag = int(profile.age_years is None)
    age_5yr = age_5yr_band(profile.age_years, profile.age_band)
    has_prior_year = int(profile.has_prior_year)
    prior_year_annual_claim_cost = float(profile.prior_year_annual_claim_cost)
    current_year_high_cost_indicator = int(profile.current_year_high_cost_indicator)
    if current_year_high_cost_indicator == 0 and profile.annual_cost_year_decile == 10:
        current_year_high_cost_indicator = 1
    prior_year_high_cost_indicator = int(profile.prior_year_high_cost_indicator)
    high_cost_last_2yr_count = current_year_high_cost_indicator + prior_year_high_cost_indicator

    return {
        "enrollment_months_count": enrollment_months,
        "full_year_enrolled_flag": int(enrollment_months == 12),
        "partial_year_enrolled_flag": int(0 < enrollment_months < 12),
        "zero_enrollment_flag": int(enrollment_months <= 0),
        "low_enrollment_flag": int(0 < enrollment_months <= 3),
        "enrollment_months_band": enrollment_months_band(enrollment_months),
        "enrollment_fraction": float(enrollment_months) / 12.0,
        "annualized_cost_per_enrolled_month": safe_rate(annual_claim_cost, enrollment_months) * 12.0,
        "annualized_claims_per_enrolled_month": safe_rate(total_claim_count, enrollment_months) * 12.0,
        "chronic_condition_count": chronic_count,
        "chronic_burden_band": burden_band,
        "age_years": int(age_years),
        "age_5yr_band": age_5yr,
        "age_missing_flag": age_missing_flag,
        "age_years_imputed": int(age_years),
        "age_over_65": max(age_years - 65, 0),
        "age_over_75": max(age_years - 75, 0),
        "age_over_85": max(age_years - 85, 0),
        "age_squared": int(age_years) * int(age_years),
        "chronic_condition_count_squared": chronic_count * chronic_count,
        "alzheimers_flag": int(profile.alzheimers_flag),
        "chf_flag": int(profile.chf_flag),
        "chronic_kidney_disease_flag": int(profile.chronic_kidney_disease_flag),
        "cancer_flag": int(profile.cancer_flag),
        "copd_flag": int(profile.copd_flag),
        "depression_flag": int(profile.depression_flag),
        "diabetes_flag": int(profile.diabetes_flag),
        "ischemic_heart_disease_flag": int(profile.ischemic_heart_disease_flag),
        "osteoporosis_flag": int(profile.osteoporosis_flag),
        "rheumatoid_arthritis_oa_flag": int(profile.rheumatoid_arthritis_oa_flag),
        "stroke_tia_flag": int(profile.stroke_tia_flag),
        "chronic_burden_age_band": f"{burden_band}__{profile.age_band}",
        "chronic_burden_age_5yr_band": f"{burden_band}__{age_5yr}",
        "sex_chronic_burden_band": f"{profile.sex}__{burden_band}",
        "claims_per_month_chronic_count_interaction": safe_rate(total_claim_count, enrollment_months) * chronic_count,
        "providers_per_month_chronic_count_interaction": safe_rate(profile.unique_provider_count, enrollment_months)
        * chronic_count,
        "age_inpatient_claim_interaction": int(age_years) * int(profile.inpatient_claim_count),
        "age_total_claim_interaction": int(age_years) * int(total_claim_count),
        "age_chronic_count_interaction": int(age_years) * chronic_count,
        "sex_male_chronic_count_interaction": chronic_count if profile.sex == "male" else 0,
        "sex_female_chronic_count_interaction": chronic_count if profile.sex == "female" else 0,
        "enrollment_months_total_claim_interaction": enrollment_months * int(total_claim_count),
        "enrollment_months_inpatient_interaction": enrollment_months * int(profile.inpatient_claim_count),
        "chronic_count_age_under_65": chronic_count if profile.age_band == "under_65" else 0,
        "chronic_count_age_65_74": chronic_count if profile.age_band == "65_74" else 0,
        "chronic_count_age_75_84": chronic_count if profile.age_band == "75_84" else 0,
        "chronic_count_age_85_plus": chronic_count if profile.age_band == "85_plus" else 0,
        "inpatient_claim_count": int(profile.inpatient_claim_count),
        "outpatient_claim_count": int(profile.outpatient_claim_count),
        "carrier_claim_count": int(profile.carrier_claim_count),
        "pde_claim_count": int(profile.pde_claim_count),
        "outpatient_ed_claim_count": int(profile.outpatient_ed_claim_count),
        "outpatient_line_count": int(profile.outpatient_line_count),
        "carrier_line_count": int(profile.carrier_line_count),
        "rx_days_supply": int(profile.rx_days_supply),
        "total_claim_days": int(profile.total_claim_days),
        "total_claim_count": int(total_claim_count),
        "unique_provider_count": int(profile.unique_provider_count),
        "cost_per_enrollment_month": safe_rate(annual_claim_cost, enrollment_months),
        "claims_per_enrollment_month": safe_rate(total_claim_count, enrollment_months),
        "claim_days_per_enrollment_month": safe_rate(profile.total_claim_days, enrollment_months),
        "providers_per_enrollment_month": safe_rate(profile.unique_provider_count, enrollment_months),
        "provider_fragmentation_index": safe_rate(profile.unique_provider_count, total_claim_count),
        "inpatient_claims_per_enrollment_month": safe_rate(profile.inpatient_claim_count, enrollment_months),
        "outpatient_claims_per_enrollment_month": safe_rate(profile.outpatient_claim_count, enrollment_months),
        "carrier_claims_per_enrollment_month": safe_rate(profile.carrier_claim_count, enrollment_months),
        "rx_fills_per_enrollment_month": safe_rate(profile.pde_claim_count, enrollment_months),
        "outpatient_ed_claims_per_enrollment_month": safe_rate(profile.outpatient_ed_claim_count, enrollment_months),
        "rx_days_supply_per_enrollment_month": safe_rate(profile.rx_days_supply, enrollment_months),
        "avg_inpatient_cost_per_claim": safe_rate(profile.inpatient_total_cost, profile.inpatient_claim_count),
        "avg_outpatient_cost_per_claim": safe_rate(profile.outpatient_total_cost, profile.outpatient_claim_count),
        "avg_carrier_cost_per_claim": safe_rate(profile.carrier_total_cost, profile.carrier_claim_count),
        "avg_rx_cost_per_fill": safe_rate(profile.rx_total_cost, profile.pde_claim_count),
        "outpatient_lines_per_claim": safe_rate(profile.outpatient_line_count, profile.outpatient_claim_count),
        "carrier_lines_per_claim": safe_rate(profile.carrier_line_count, profile.carrier_claim_count),
        "any_inpatient_claim": int(profile.inpatient_claim_count > 0),
        "any_outpatient_claim": int(profile.outpatient_claim_count > 0),
        "any_carrier_claim": int(profile.carrier_claim_count > 0),
        "any_pde_claim": int(profile.pde_claim_count > 0),
        "any_outpatient_ed_claim": int(profile.outpatient_ed_claim_count > 0),
        "multiple_provider_flag": int(profile.unique_provider_count > 1),
        "multi_setting_utilization_flag": int(
            sum(
                [
                    profile.inpatient_claim_count > 0,
                    profile.outpatient_claim_count > 0,
                    profile.carrier_claim_count > 0,
                    profile.pde_claim_count > 0,
                ]
            )
            >= 2
        ),
        "rx_total_cost": float(profile.rx_total_cost),
        "inpatient_total_cost": float(profile.inpatient_total_cost),
        "outpatient_total_cost": float(profile.outpatient_total_cost),
        "carrier_total_cost": float(profile.carrier_total_cost),
        "carrier_allowed_total": float(profile.carrier_allowed_total),
        "rx_patient_pay_total": float(profile.rx_patient_pay_total),
        "rx_cost_log1p": nonnegative_log1p(profile.rx_total_cost),
        "inpatient_cost_log1p": nonnegative_log1p(profile.inpatient_total_cost),
        "outpatient_cost_log1p": nonnegative_log1p(profile.outpatient_total_cost),
        "carrier_cost_log1p": nonnegative_log1p(profile.carrier_total_cost),
        "annual_cost_log1p": nonnegative_log1p(annual_claim_cost),
        "inpatient_claim_count_log1p": nonnegative_log1p(profile.inpatient_claim_count),
        "outpatient_claim_count_log1p": nonnegative_log1p(profile.outpatient_claim_count),
        "carrier_claim_count_log1p": nonnegative_log1p(profile.carrier_claim_count),
        "pde_claim_count_log1p": nonnegative_log1p(profile.pde_claim_count),
        "total_claim_count_log1p": nonnegative_log1p(total_claim_count),
        "unique_provider_count_log1p": nonnegative_log1p(profile.unique_provider_count),
        "annual_cost_year_percentile": float(profile.annual_cost_year_percentile),
        "annual_cost_year_decile": int(profile.annual_cost_year_decile),
        "annual_cost_to_year_median": float(profile.annual_cost_to_year_median),
        "has_prior_year": has_prior_year,
        "prior_year_annual_claim_cost": prior_year_annual_claim_cost,
        "prior_year_inpatient_claim_count": int(profile.prior_year_inpatient_claim_count),
        "prior_year_total_claim_count": int(profile.prior_year_total_claim_count),
        "prior_year_enrollment_months_count": int(profile.prior_year_enrollment_months_count),
        "current_year_high_cost_indicator": current_year_high_cost_indicator,
        "prior_year_high_cost_indicator": prior_year_high_cost_indicator,
        "two_year_avg_annual_claim_cost": (
            annual_claim_cost + (prior_year_annual_claim_cost if has_prior_year else 0.0)
        )
        / (has_prior_year + 1),
        "cost_trend_difference": annual_claim_cost - prior_year_annual_claim_cost,
        "cost_trend_ratio": safe_rate(annual_claim_cost, prior_year_annual_claim_cost),
        "inpatient_claim_count_change": int(profile.inpatient_claim_count)
        - int(profile.prior_year_inpatient_claim_count),
        "total_claim_count_change": int(total_claim_count) - int(profile.prior_year_total_claim_count),
        "utilization_trend": utilization_trend(total_claim_count, profile.prior_year_total_claim_count, has_prior_year),
        "high_cost_last_2yr_count": high_cost_last_2yr_count,
        "high_cost_1_of_last_2yr": int(high_cost_last_2yr_count >= 1),
        "high_cost_2_of_last_2yr": int(high_cost_last_2yr_count >= 2),
        "inpatient_cost_share": safe_rate(profile.inpatient_total_cost, positive_cost_total),
        "outpatient_cost_share": safe_rate(profile.outpatient_total_cost, positive_cost_total),
        "carrier_cost_share": safe_rate(profile.carrier_total_cost, positive_cost_total),
        "rx_cost_share": safe_rate(profile.rx_total_cost, positive_cost_total),
        "age_band": profile.age_band,
        "sex": profile.sex,
        "race_code": profile.race_code,
        "state_code": profile.state_code,
    }


def expected_model_feature_order(model=None) -> list[str]:
    if model is not None:
        signature_order = deterministic_feature_order(model)
        if signature_order:
            return signature_order
    return SERVED_MODEL_FEATURE_ORDER


def build_model_frame(feature_row: dict[str, float | int | str], model=None) -> pd.DataFrame:
    features = pd.DataFrame([feature_row])
    for column in MODEL_INT32_FIELDS:
        if column in features.columns:
            features[column] = features[column].astype(np.int32)
    for column in MODEL_INT64_FIELDS:
        if column in features.columns:
            features[column] = features[column].astype(np.int64)
    feature_order = expected_model_feature_order(model)
    missing_columns = [column for column in feature_order if column not in features.columns]
    if missing_columns:
        raise ValueError(f"Missing engineered model features: {', '.join(missing_columns)}")
    return features[feature_order]


def _final_estimator(model):
    if hasattr(model, "steps") and model.steps:
        return model.steps[-1][1]
    return model


def extract_model_probabilities(model, features: pd.DataFrame) -> tuple[float, float]:
    calibrated_probability = float(model.predict_proba(features)[0][1])
    estimator = _final_estimator(model)
    if not hasattr(estimator, "calibrated_classifiers_"):
        return calibrated_probability, calibrated_probability

    raw_estimator = getattr(estimator, "estimator", None) or getattr(estimator, "base_estimator", None)
    if raw_estimator is None:
        return calibrated_probability, calibrated_probability

    try:
        raw_features = features
        if hasattr(model, "steps") and len(model.steps) > 1:
            raw_features = model[:-1].transform(features)
        raw_model_probability = float(raw_estimator.predict_proba(raw_features)[0][1])
        return raw_model_probability, calibrated_probability
    except Exception:
        return calibrated_probability, calibrated_probability


def build_prediction_response(profile: BeneficiaryProfile) -> PredictionResponse:
    feature_row = build_feature_row(profile)
    model = load_model()
    features = build_model_frame(feature_row, model)
    metadata = load_model_metadata()
    decision_threshold = float(metadata.get("decision_threshold", MODEL_DECISION_THRESHOLD))
    threshold_source = str(metadata.get("decision_threshold_source", "validation_f1"))
    raw_model_probability, calibrated_probability = extract_model_probabilities(model, features)

    reference_distribution = load_reference_distribution()
    score_0_100, transform = compute_risk_score(calibrated_probability, reference_distribution)
    tier = assign_risk_tier(score_0_100)
    reasons = generate_reason_codes(feature_row, metadata.get("reason_thresholds", {}))
    model_name = str(metadata.get("model_name", MODEL_NAME))
    model_version = str(metadata.get("model_version", MODEL_VERSION))
    feature_contract_version = str(metadata.get("feature_contract_version", FEATURE_CONTRACT_VERSION))
    calibration_method = str(metadata.get("calibration_method", "unknown"))
    split_version = str(metadata.get("split_version", "unknown"))
    intervention_flag = calibrated_probability >= decision_threshold
    
    annual_claim_cost_proxy = (
        float(profile.rx_total_cost)
        + float(profile.inpatient_total_cost)
        + float(profile.outpatient_total_cost)
        + float(profile.carrier_total_cost)
    )
    cost_mix = {
        "inpatient": float(profile.inpatient_total_cost),
        "outpatient": float(profile.outpatient_total_cost),
        "carrier": float(profile.carrier_total_cost),
        "prescription": float(profile.rx_total_cost),
    }
    
    metrics = PredictionMetrics(
        raw_model_probability=raw_model_probability,
        calibrated_probability=calibrated_probability,
        risk_score_0_100=score_0_100,
        risk_tier=tier,
        intervention_flag=intervention_flag,
        decision_threshold=decision_threshold,
    )
    
    prediction_metadata = PredictionMetadata(
        model_name=model_name,
        feature_contract_version=feature_contract_version,
        calibration_method=calibration_method,
        threshold_source=threshold_source,
        split_version=split_version,
        score_transform=transform,
    )
    
    return PredictionResponse(
        model_name=model_name,
        model_version=model_version,
        raw_model_probability=raw_model_probability,
        calibrated_probability=calibrated_probability,
        risk_score_0_100=score_0_100,
        risk_tier=tier,
        predicted_high_cost=intervention_flag,
        intervention_flag=intervention_flag,
        decision_threshold=decision_threshold,
        threshold_source=threshold_source,
        reason_codes=reasons,
        feature_contract_version=feature_contract_version,
        calibration_method=calibration_method,
        split_version=split_version,
        score_transform=transform,
        reference_distribution_available=bool(reference_distribution),
        risk_probability=calibrated_probability,
        risk_score=calibrated_probability,
        prediction=metrics,
        metadata=prediction_metadata,
        annual_claim_cost_proxy=annual_claim_cost_proxy,
        cost_mix=cost_mix,
        engineered_features=feature_row,
    )


def build_state_response(profile: BeneficiaryProfile) -> StateResponse:
    feature_row = build_feature_row(profile)
    probability, current_state = profile_to_state(feature_row, profile, load_model())
    return StateResponse(
        risk_probability=probability,
        current_state=current_state,
        disclaimer=SIMULATION_DISCLAIMER,
    )


def build_recommendation_response(profile: BeneficiaryProfile) -> RecommendationResponse:
    feature_row = build_feature_row(profile)
    recommendation = action_values_for_profile(feature_row, profile, load_model(), load_q_table_artifact())
    return RecommendationResponse(
        risk_probability=float(recommendation["risk_probability"]),
        risk_tier=str(recommendation["risk_tier"]),
        current_state=dict(recommendation["state"]),
        recommended_action=str(recommendation["recommended_action"]),
        recommended_action_label=str(recommendation["recommended_action_label"]),
        recommended_action_display=str(recommendation["recommended_action_display"]),
        expected_long_run_value=float(recommendation["expected_long_run_value"]),
        q_values={str(key): float(value) for key, value in recommendation["q_values"].items()},
        action_values=[ActionValueResponse(**value) for value in recommendation["action_values"]],
        policy_explanation=str(recommendation["policy_explanation"]),
        disclaimer=str(recommendation["disclaimer"]),
    )


def build_simulation_response(profile: BeneficiaryProfile) -> SimulationResponse:
    feature_row = build_feature_row(profile)
    comparison = compare_all_actions(feature_row, profile, load_model(), load_q_table_artifact())
    return SimulationResponse(
        baseline_risk_probability=float(comparison["baseline_risk_probability"]),
        risk_tier=str(comparison["risk_tier"]),
        current_state=dict(comparison["state"]),
        recommended_action=str(comparison["recommended_action"]),
        recommended_action_label=str(comparison["recommended_action_label"]),
        recommended_action_display=str(comparison["recommended_action_display"]),
        q_values={str(key): float(value) for key, value in comparison["q_values"].items()},
        comparisons=[SimulationComparisonResponse(**value) for value in comparison["comparisons"]],
        policy_explanation=str(comparison["policy_explanation"]),
        disclaimer=str(comparison["disclaimer"]),
    )


def install_sklearn_compat_shim() -> None:
    """Provide the legacy sklearn gradient-boosting losses module for older pickles."""

    if "sklearn.ensemble._gb_losses" in sys.modules:
        return

    from sklearn.tree import DecisionTreeRegressor

    module = types.ModuleType("sklearn.ensemble._gb_losses")
    if not hasattr(DecisionTreeRegressor, "monotonic_cst"):
        DecisionTreeRegressor.monotonic_cst = None

    class _HalfBinomialLogitLink:
        @staticmethod
        def link(predictions):
            predictions = np.asarray(predictions, dtype=float)
            clipped = np.clip(predictions, 1e-7, 1.0 - 1e-7)
            return 0.5 * np.log(clipped / (1.0 - clipped))

    class _MultinomialLogitLink:
        @staticmethod
        def link(predictions):
            predictions = np.asarray(predictions, dtype=float)
            clipped = np.clip(predictions, 1e-7, 1.0)
            return np.log(clipped)

    class _BaseLoss:
        def __init__(self, *args, **kwargs):
            self.K = kwargs.get("K", 1)

    class BinomialDeviance(_BaseLoss):
        is_multiclass = False
        link = _HalfBinomialLogitLink()

        def predict_proba(self, raw_predictions):
            scores = np.asarray(raw_predictions, dtype=float).reshape(-1, 1)
            positive = 1.0 / (1.0 + np.exp(-2.0 * scores[:, 0]))
            negative = 1.0 - positive
            return np.column_stack([negative, positive])

        def get_init_raw_predictions(self, X, estimator):
            probas = estimator.predict_proba(X)
            return self.link.link(probas[:, 1]).reshape(-1, 1)

    class MultinomialDeviance(_BaseLoss):
        is_multiclass = True
        link = _MultinomialLogitLink()

        def predict_proba(self, raw_predictions):
            scores = np.asarray(raw_predictions, dtype=float)
            scores = scores - np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    class LeastSquaresError(_BaseLoss):
        pass

    class LeastAbsoluteError(_BaseLoss):
        pass

    class HuberLossFunction(_BaseLoss):
        pass

    class QuantileLossFunction(_BaseLoss):
        pass

    module.BinomialDeviance = BinomialDeviance
    module.MultinomialDeviance = MultinomialDeviance
    module.LeastSquaresError = LeastSquaresError
    module.LeastAbsoluteError = LeastAbsoluteError
    module.HuberLossFunction = HuberLossFunction
    module.QuantileLossFunction = QuantileLossFunction
    sys.modules["sklearn.ensemble._gb_losses"] = module


@lru_cache(maxsize=1)
def load_model():
    install_sklearn_compat_shim()
    return mlflow.sklearn.load_model(str(MODEL_PATH))


@lru_cache(maxsize=1)
def load_q_table_artifact() -> np.ndarray:
    return load_q_table(Q_TABLE_PATH)


@lru_cache(maxsize=1)
def load_rl_metadata() -> dict[str, object]:
    return json.loads(RL_METADATA_PATH.read_text())


app = FastAPI(
    title="Actuarial Decision-Support API",
    version="2.0.0",
    description=(
        "Serves next-year Medicare high-cost risk predictions from the gradient-boosting model "
        "and simulated MDP/Q-learning intervention recommendations."
    ),
)


@app.get("/health")
def health() -> dict[str, str]:
    load_model()
    load_q_table_artifact()
    metadata = load_model_metadata()
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "q_table_path": str(Q_TABLE_PATH),
        "model_name": str(metadata["model_name"]),
        "model_version": str(metadata["model_version"]),
        "sklearn_version": str(metadata["sklearn_version"]),
    }


@app.get("/metadata")
def metadata() -> dict[str, object]:
    return {
        "risk_engine": {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "model_metadata": load_model_metadata(),
            "prediction_target": "next_year_high_cost",
            "decision_threshold": MODEL_DECISION_THRESHOLD,
            "operating_policy": "risk_score_0_100 >= 95 very high; >= 90 high; >= 75 elevated; otherwise low",
            "required_fields": list(BeneficiaryProfile.model_fields.keys()),
        },
        "policy_layer": load_rl_metadata(),
        "disclaimer": SIMULATION_DISCLAIMER,
    }


@app.get("/model_metrics")
def model_metrics() -> dict[str, object]:
    if MODEL_METRICS_PATH.exists():
        return json.loads(MODEL_METRICS_PATH.read_text())
    return {}


@app.post("/predict", response_model=PredictionResponse)
def predict(profile: BeneficiaryProfile) -> PredictionResponse:
    return build_prediction_response(profile)


@app.post("/state", response_model=StateResponse)
def derive_state(profile: BeneficiaryProfile) -> StateResponse:
    return build_state_response(profile)


@app.post("/recommend_action", response_model=RecommendationResponse)
def recommend_action(profile: BeneficiaryProfile) -> RecommendationResponse:
    return build_recommendation_response(profile)


@app.post("/simulate", response_model=SimulationResponse)
def simulate(profile: BeneficiaryProfile) -> SimulationResponse:
    return build_simulation_response(profile)


@app.post("/decision_support", response_model=DecisionSupportResponse)
def decision_support(profile: BeneficiaryProfile) -> DecisionSupportResponse:
    prediction = build_prediction_response(profile)
    state = build_state_response(profile)
    recommendation = build_recommendation_response(profile)
    simulation = build_simulation_response(profile)
    return DecisionSupportResponse(
        prediction=prediction,
        state=state,
        recommendation=recommendation,
        simulation=simulation,
        disclaimer=SIMULATION_DISCLAIMER,
    )
