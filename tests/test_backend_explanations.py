from __future__ import annotations

from backend.explanations import reason_code_version, top_risk_drivers


def test_prediction_response_contains_reason_codes() -> None:
    drivers = top_risk_drivers(
        {
            "chronic_condition_count": 8,
            "claims_per_enrollment_month": 2.5,
            "provider_fragmentation_index": 0.8,
            "total_claim_days": 35,
            "inpatient_claim_count": 1,
            "unique_provider_count": 9,
        }
    )
    assert "high chronic condition count" in drivers
    assert "prior inpatient utilization" in drivers
    assert reason_code_version() == "reason_codes_v1"
