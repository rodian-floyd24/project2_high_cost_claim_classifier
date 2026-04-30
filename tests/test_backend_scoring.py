from __future__ import annotations

from backend.scoring import MODEL_NAME, MODEL_VERSION, operating_policy_for_score, score_to_operating_tier


def test_operating_tier_is_deterministic() -> None:
    assert score_to_operating_tier(0.30) == "very_high"
    assert score_to_operating_tier(0.20) == "high"
    assert score_to_operating_tier(0.10) == "elevated"
    assert score_to_operating_tier(0.099) == "routine"


def test_operating_policy_returns_action() -> None:
    tier, operating_risk_score, action = operating_policy_for_score(0.22)
    assert tier == "high"
    assert operating_risk_score == 0.22
    assert action == "moderate_outreach"


def test_model_metadata_constants_are_present() -> None:
    assert MODEL_NAME
    assert MODEL_VERSION
