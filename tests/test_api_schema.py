from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import app


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


def test_invalid_negative_cost_is_rejected() -> None:
    payload = dict(VALID_PAYLOAD)
    payload["inpatient_total_cost"] = -1.0
    response = TestClient(app).post("/predict", json=payload)
    assert response.status_code == 422


def test_invalid_enrollment_months_rejected() -> None:
    payload = dict(VALID_PAYLOAD)
    payload["enrollment_months_count"] = 13
    response = TestClient(app).post("/predict", json=payload)
    assert response.status_code == 422


def test_valid_profile_scores_successfully() -> None:
    response = TestClient(app).post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200, response.text


def test_prediction_response_contains_governance_fields() -> None:
    response = TestClient(app).post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_version"]
    assert body["risk_score"] == body["risk_probability"]
    assert body["operating_risk_score"] == body["risk_probability"]
    assert "risk_percentile" not in body
    assert body["recommended_action"]
    assert body["top_risk_drivers"]
    assert "input_review_flags" in body
    assert "human_review_required" in body
    assert body["reason_code_version"] == "reason_codes_v1"

def test_metadata_endpoint_contains_calibration_fields() -> None:
    response = TestClient(app).get("/metadata")
    assert response.status_code == 200, response.text
    body = response.json()
    model_metadata = body["risk_engine"]["model_metadata"]
    assert "calibration_method" in model_metadata
    assert "calibration_status" in model_metadata
    assert "probability_interpretation" in model_metadata
    assert "ranking_use_statement" in model_metadata
