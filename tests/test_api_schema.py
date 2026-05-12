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
    assert set(body) >= {
        "model_name",
        "model_version",
        "raw_model_probability",
        "calibrated_probability",
        "risk_score_0_100",
        "risk_tier",
        "predicted_high_cost",
        "intervention_flag",
        "decision_threshold",
        "threshold_source",
        "prediction",
        "reason_codes",
        "feature_contract_version",
        "calibration_method",
        "split_version",
        "score_transform",
        "reference_distribution_available",
        "risk_probability",
        "risk_score",
        "metadata",
        "annual_claim_cost_proxy",
        "cost_mix",
        "engineered_features",
    }
    prediction = body["prediction"]
    assert body["raw_model_probability"] == prediction["raw_model_probability"]
    assert body["calibrated_probability"] == prediction["calibrated_probability"]
    assert body["risk_score_0_100"] == prediction["risk_score_0_100"]
    assert body["risk_tier"] == prediction["risk_tier"]
    assert body["intervention_flag"] == prediction["intervention_flag"]
    assert body["decision_threshold"] == prediction["decision_threshold"]
    assert body["predicted_high_cost"] == prediction["intervention_flag"]
    assert body["risk_probability"] == prediction["calibrated_probability"]
    assert body["risk_score"] == prediction["calibrated_probability"]
    assert body["risk_score_0_100"] != int(round(body["risk_probability"] * 100))
    assert 0.0 <= prediction["raw_model_probability"] <= 1.0
    assert 0.0 <= prediction["calibrated_probability"] <= 1.0
    assert 0 <= prediction["risk_score_0_100"] <= 100
    assert prediction["risk_tier"] in {"low", "elevated", "high", "very_high"}
    assert isinstance(prediction["intervention_flag"], bool)
    assert body["reason_codes"]
    assert body["metadata"]["model_name"]
    assert body["metadata"]["calibration_method"]
    assert body["metadata"]["score_transform"]

def test_metadata_endpoint_contains_calibration_fields() -> None:
    response = TestClient(app).get("/metadata")
    assert response.status_code == 200, response.text
    body = response.json()
    model_metadata = body["risk_engine"]["model_metadata"]
    assert "calibration_method" in model_metadata
    assert "calibration_status" in model_metadata
    assert "probability_interpretation" in model_metadata
    assert "ranking_use_statement" in model_metadata
