from __future__ import annotations

from backend.scoring import (
    CALIBRATION_METHOD,
    CALIBRATION_STATUS,
    MODEL_NAME,
    MODEL_VERSION,
    PROBABILITY_INTERPRETATION,
    RANKING_USE_STATEMENT,
    assign_risk_tier,
    compute_risk_score,
    generate_reason_codes,
)


def test_operating_tier_is_deterministic() -> None:
    assert assign_risk_tier(95) == "very_high"
    assert assign_risk_tier(90) == "high"
    assert assign_risk_tier(75) == "elevated"
    assert assign_risk_tier(74) == "low"


def test_risk_score_transform_returns_display_score() -> None:
    score, transform = compute_risk_score(0.047)
    assert 0 <= score <= 100
    assert transform == "monotone_probability_fallback_v1"


def test_reason_codes_return_top_drivers() -> None:
    reasons = generate_reason_codes(
        {
            "chronic_condition_count": 7,
            "claims_per_enrollment_month": 3.2,
            "provider_fragmentation_index": 0.8,
            "any_carrier_claim": 1,
            "total_claim_days_log1p": 3.4,
        },
        {
            "claims_per_enrollment_month": 3.0,
            "provider_fragmentation_index": 0.75,
            "total_claim_days_log1p": 3.0,
        },
    )
    assert reasons == [
        "High chronic condition burden",
        "High claims per enrollment month",
        "High provider fragmentation",
    ]


def test_model_metadata_constants_are_present() -> None:
    assert MODEL_NAME
    assert MODEL_VERSION
    assert CALIBRATION_METHOD
    assert CALIBRATION_STATUS
    assert PROBABILITY_INTERPRETATION
    assert RANKING_USE_STATEMENT
