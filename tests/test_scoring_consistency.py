from __future__ import annotations

from backend.explanations import top_risk_drivers
from backend.scoring import assign_risk_tier, compute_risk_score


def test_score_to_operating_tier_is_deterministic() -> None:
    assert assign_risk_tier(99) == "very_high"
    assert assign_risk_tier(92) == "high"
    assert assign_risk_tier(80) == "elevated"
    assert assign_risk_tier(50) == "low"
    score, transform = compute_risk_score(0.31)
    assert score > 90
    assert transform == "monotone_probability_fallback_v1"


def test_reason_codes_are_stable() -> None:
    feature_row = {
        "chronic_condition_count": 7,
        "inpatient_claim_count": 2,
        "total_claim_count": 24,
        "claims_per_enrollment_month": 2.0,
        "unique_provider_count": 8,
        "prior_year_high_cost_indicator": 1,
        "cost_per_enrollment_month": 1200.0,
        "enrollment_months_count": 12,
    }
    drivers = top_risk_drivers(feature_row)
    assert drivers[0] == "high chronic condition count"
    assert len(drivers) == 5
