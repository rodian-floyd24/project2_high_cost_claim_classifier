from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import app, build_model_frame
from backend.scoring import assign_risk_tier, compute_risk_score, load_reference_distribution
from shared.feature_contract import FEATURE_CONTRACT_VERSION, SERVED_MODEL_FEATURE_ORDER


VALID_PAYLOAD = {
    "age_band": "75_84",
    "sex": "male",
    "race_code": "1",
    "state_code": "TX",
    "enrollment_months_count": 12,
    "chronic_condition_count": 4,
    "alzheimers_flag": True,
    "chf_flag": True,
    "diabetes_flag": True,
    "ischemic_heart_disease_flag": True,
    "inpatient_claim_count": 1,
    "outpatient_claim_count": 8,
    "carrier_claim_count": 12,
    "pde_claim_count": 20,
    "total_claim_days": 11,
    "unique_provider_count": 7,
    "rx_total_cost": 3200.0,
    "inpatient_total_cost": 6400.0,
    "outpatient_total_cost": 2800.0,
    "carrier_total_cost": 2400.0,
    "prior_intervention_status": "recent_low_touch",
}


def score_payload() -> dict:
    response = TestClient(app).post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200, response.text
    return response.json()


def test_live_score_response_contains_required_fields() -> None:
    body = score_payload()
    assert set(body) >= {
        "model_name",
        "model_version",
        "raw_model_probability",
        "calibrated_probability",
        "risk_score_0_100",
        "risk_tier",
        "predicted_high_cost",
        "decision_threshold",
        "threshold_source",
        "reason_codes",
        "feature_contract_version",
        "calibration_method",
        "split_version",
        "score_transform",
        "reference_distribution_available",
        "risk_probability",
        "risk_score",
    }


def test_risk_score_between_0_and_100() -> None:
    body = score_payload()
    assert 0 <= body["risk_score_0_100"] <= 100


def test_risk_tier_matches_score_cutoffs() -> None:
    assert assign_risk_tier(95) == "very_high"
    assert assign_risk_tier(90) == "high"
    assert assign_risk_tier(75) == "elevated"
    assert assign_risk_tier(74) == "low"
    body = score_payload()
    assert body["risk_tier"] == assign_risk_tier(body["risk_score_0_100"])


def test_raw_probability_not_used_as_display_score() -> None:
    body = score_payload()
    raw_probability_as_score = int(round(body["raw_model_probability"] * 100))
    assert body["risk_score_0_100"] != raw_probability_as_score
    assert body["risk_score"] == body["calibrated_probability"]
    assert body["risk_probability"] == body["calibrated_probability"]


def test_calibrated_probability_present_when_calibrator_available() -> None:
    body = score_payload()
    assert "calibrated_probability" in body
    assert 0.0 <= body["calibrated_probability"] <= 1.0


def test_feature_contract_used_by_serving() -> None:
    body = score_payload()
    assert body["feature_contract_version"] == FEATURE_CONTRACT_VERSION
    frame = build_model_frame(body["engineered_features"])
    assert list(frame.columns) == SERVED_MODEL_FEATURE_ORDER


def test_decision_threshold_not_default_point_five_unless_explicitly_justified() -> None:
    body = score_payload()
    assert body["decision_threshold"] != 0.5
    assert body["threshold_source"]


def test_metadata_contains_calibration_and_threshold_policy() -> None:
    body = score_payload()
    assert body["calibration_method"]
    assert body["threshold_source"]
    assert body["score_transform"]
    assert isinstance(body["reference_distribution_available"], bool)


def test_reason_codes_are_bounded_to_top_three() -> None:
    body = score_payload()
    assert 1 <= len(body["reason_codes"]) <= 3


def test_reference_distribution_drives_percentile_score() -> None:
    reference_distribution = load_reference_distribution()
    assert reference_distribution
    score, transform = compute_risk_score(0.0641, reference_distribution)
    assert 0 <= score <= 100
    assert transform == "percentile_rank_calibrated_probability"
