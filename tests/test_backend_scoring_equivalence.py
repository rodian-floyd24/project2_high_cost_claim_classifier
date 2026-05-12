from __future__ import annotations

import math

import numpy as np
import pytest

from backend.app import BeneficiaryProfile, build_feature_row, build_model_frame
from shared.feature_contract import MODEL_FEATURE_ORDER


def test_build_model_frame_matches_served_feature_order() -> None:
    profile = BeneficiaryProfile(
        age_band="65_74",
        sex="female",
        race_code="white",
        state_code="CA",
        enrollment_months_count=12,
        chronic_condition_count=3,
        inpatient_claim_count=1,
        outpatient_claim_count=2,
        carrier_claim_count=5,
        pde_claim_count=10,
        total_claim_days=15,
        unique_provider_count=4,
        rx_total_cost=500.0,
        inpatient_total_cost=5000.0,
        outpatient_total_cost=1000.0,
        carrier_total_cost=2000.0,
    )
    feature_row = build_feature_row(profile)
    # Don't pass model; it should default to the canonical model feature order.
    features = build_model_frame(feature_row)
    assert list(features.columns) == MODEL_FEATURE_ORDER


def test_numerical_feature_formulas() -> None:
    profile = BeneficiaryProfile(
        age_band="65_74",
        sex="female",
        race_code="white",
        state_code="CA",
        enrollment_months_count=6,
        chronic_condition_count=3,
        inpatient_claim_count=1,
        outpatient_claim_count=2,
        carrier_claim_count=5,
        pde_claim_count=10,
        total_claim_days=15,
        unique_provider_count=4,
        rx_total_cost=500.0,
        inpatient_total_cost=5000.0,
        outpatient_total_cost=1000.0,
        carrier_total_cost=2000.0,
    )
    feature_row = build_feature_row(profile)

    expected_total_claim_count = 1 + 2 + 5 + 10
    expected_annual_claim_cost = 500.0 + 5000.0 + 1000.0 + 2000.0
    expected_cost_per_enrollment_month = expected_annual_claim_cost / 6.0
    expected_claims_per_enrollment_month = expected_total_claim_count / 6.0
    expected_provider_fragmentation_index = 4.0 / expected_total_claim_count
    expected_annual_cost_log1p = math.log1p(expected_annual_claim_cost)

    positive_cost_total = max(500.0, 0) + max(5000.0, 0) + max(1000.0, 0) + max(2000.0, 0)
    expected_inpatient_cost_share = 5000.0 / positive_cost_total
    expected_outpatient_cost_share = 1000.0 / positive_cost_total
    expected_carrier_cost_share = 2000.0 / positive_cost_total
    expected_rx_cost_share = 500.0 / positive_cost_total

    assert feature_row["total_claim_count"] == expected_total_claim_count
    assert np.isclose(feature_row["cost_per_enrollment_month"], expected_cost_per_enrollment_month)
    assert np.isclose(feature_row["claims_per_enrollment_month"], expected_claims_per_enrollment_month)
    assert np.isclose(feature_row["provider_fragmentation_index"], expected_provider_fragmentation_index)
    assert np.isclose(feature_row["annual_cost_log1p"], expected_annual_cost_log1p)
    assert np.isclose(feature_row["inpatient_cost_share"], expected_inpatient_cost_share)
    assert np.isclose(feature_row["outpatient_cost_share"], expected_outpatient_cost_share)
    assert np.isclose(feature_row["carrier_cost_share"], expected_carrier_cost_share)
    assert np.isclose(feature_row["rx_cost_share"], expected_rx_cost_share)
