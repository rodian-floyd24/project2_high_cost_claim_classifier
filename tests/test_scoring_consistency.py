from __future__ import annotations

from backend.explanations import top_risk_drivers
from backend.scoring import operating_policy_for_score, score_to_operating_tier


def test_score_to_operating_tier_is_deterministic() -> None:
    assert score_to_operating_tier(0.31) == "very_high"
    assert score_to_operating_tier(0.21) == "high"
    assert score_to_operating_tier(0.11) == "elevated"
    assert score_to_operating_tier(0.09) == "routine"
    assert operating_policy_for_score(0.31)[2] == "intensive_case_management_review"


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
