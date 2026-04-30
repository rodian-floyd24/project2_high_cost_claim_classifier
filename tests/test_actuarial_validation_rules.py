from __future__ import annotations

import pytest

from backend.schemas import input_review_flags, validate_profile_values


def test_invalid_negative_claim_count_rejected() -> None:
    with pytest.raises(ValueError):
        validate_profile_values({"inpatient_claim_count": -1})


def test_invalid_negative_cost_rejected() -> None:
    with pytest.raises(ValueError):
        validate_profile_values({"rx_total_cost": -0.01})


def test_input_review_flags_identify_partial_year_and_extreme_values() -> None:
    flags = input_review_flags(
        {
            "enrollment_months_count": 5,
            "annualized_cost_per_enrolled_month": 300_000.0,
            "total_claim_count": 1_001,
            "provider_fragmentation_index": 1.2,
        }
    )
    assert "partial_year_enrollment" in flags
    assert "extreme_annualized_cost" in flags
    assert "extreme_claim_count" in flags
    assert "provider_count_exceeds_claim_count" in flags
