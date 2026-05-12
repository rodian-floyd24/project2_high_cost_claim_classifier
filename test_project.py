"""
Test Script for Actuarial Decision-Support Prototype
====================================================
Sends a sample beneficiary profile to the prediction endpoint and prints the result.

Default behavior:
- If PROJECT2_API_URL is set, sends HTTP requests to that deployed API.
- Otherwise, runs the FastAPI app in-process with TestClient and validates the
  same endpoints locally without requiring a running server.

Requirements:
- Local fallback: `pip install -r requirements-dev.txt`
- Deployed API mode: `pip install -r requirements-dev.txt`

Usage:
- `python test_project.py`
- `PROJECT2_API_URL=https://your-app-url.com python test_project.py`
"""

from __future__ import annotations

import json
import math
import os
import sys
import warnings
from typing import Any

from sklearn.exceptions import InconsistentVersionWarning


SAMPLE_PAYLOAD = {
    "age_band": "75_84",
    "sex": "male",
    "race_code": "1",
    "state_code": "TX",
    "enrollment_months_count": 12,
    "chronic_condition_count": 4,
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

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


def probability_to_display_score(probability: float, scale: float = 0.10) -> int:
    probability = max(0.0, min(float(probability), 1.0))
    score = 100.0 * (1.0 - math.exp(-probability / scale))
    return int(round(max(0.0, min(score, 100.0))))


def normalized_prediction_metrics(result: dict[str, Any]) -> dict[str, Any]:
    if "prediction" in result and isinstance(result["prediction"], dict):
        metrics = result["prediction"]
        tier = str(metrics["risk_tier"]).strip().lower().replace(" ", "_")
        return {
            "raw_model_probability": float(metrics["raw_model_probability"]),
            "calibrated_probability": float(metrics["calibrated_probability"]),
            "risk_score_0_100": int(metrics["risk_score_0_100"]),
            "risk_tier": tier,
            "intervention_flag": bool(metrics["intervention_flag"]),
            "decision_threshold": float(metrics["decision_threshold"]),
            "schema": "nested",
        }

    if "risk_probability" in result:
        probability = float(result["risk_probability"])
        threshold = float(result.get("decision_threshold", 0.20))
        tier = str(result["risk_tier"]).strip().lower().replace(" ", "_")
        return {
            "raw_model_probability": probability,
            "calibrated_probability": probability,
            "risk_score_0_100": int(result.get("risk_score_0_100", probability_to_display_score(probability))),
            "risk_tier": tier,
            "intervention_flag": bool(result.get("predicted_high_cost", probability >= threshold)),
            "decision_threshold": threshold,
            "schema": "legacy_flat",
        }

    raise RuntimeError("Response does not contain a recognized prediction schema")


def print_summary(result: dict[str, Any], mode: str) -> None:
    metrics = normalized_prediction_metrics(result)

    print(f"Mode: {mode}")
    print("Health check: PASS")
    print("Prediction endpoint: PASS")
    print(f"Risk score returned: {metrics['risk_score_0_100']}")
    print(f"Risk tier returned: {metrics['risk_tier']}")
    print(f"Calibrated probability returned: {metrics['calibrated_probability']:.4f}")
    print(f"Intervention flag returned: {'yes' if metrics['intervention_flag'] else 'no'}")
    print("Decision-support endpoint: PASS")
    print("End-to-end test: PASS")


def validate_prediction_response(result: dict[str, Any]) -> None:
    if "prediction" in result:
        required_top_level = {
            "prediction",
            "reason_codes",
            "metadata",
            "annual_claim_cost_proxy",
            "cost_mix",
            "engineered_features",
        }
        missing_top_level = required_top_level - set(result)
        if missing_top_level:
            raise RuntimeError(f"Missing prediction response keys: {sorted(missing_top_level)}")

    prediction = normalized_prediction_metrics(result)

    if not 0.0 <= float(prediction["calibrated_probability"]) <= 1.0:
        raise RuntimeError("calibrated_probability outside [0, 1]")
    if not 0 <= int(prediction["risk_score_0_100"]) <= 100:
        raise RuntimeError("risk_score_0_100 outside [0, 100]")
    if not prediction["risk_tier"]:
        raise RuntimeError("risk_tier missing")
    if not isinstance(prediction["intervention_flag"], bool):
        raise RuntimeError("intervention_flag is not boolean")


def validate_decision_support_response(result: dict[str, Any]) -> None:
    required_top_level = {"prediction", "state", "recommendation", "simulation", "disclaimer"}
    missing = required_top_level - set(result)
    if missing:
        raise RuntimeError(f"Missing top-level keys: {sorted(missing)}")

    prediction_response = result["prediction"]
    validate_prediction_response(prediction_response)
    recommendation = result["recommendation"]
    state = result["state"]["current_state"]

    if not recommendation["recommended_action"]:
        raise RuntimeError("recommended_action missing")
    if "label" not in state:
        raise RuntimeError("state label missing")


def run_against_http(api_base_url: str) -> None:
    import requests

    base_url = api_base_url.rstrip("/")
    health_url = base_url + "/health"
    response = requests.get(health_url, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"FAIL: {health_url} returned {response.status_code}: {response.text[:500]}")
    if response.json().get("status") != "ok":
        raise RuntimeError(f"FAIL: {health_url} did not return status=ok")

    prediction_url = base_url + "/predict"
    response = requests.post(prediction_url, json=SAMPLE_PAYLOAD, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"FAIL: {prediction_url} returned {response.status_code}: {response.text[:500]}")
    prediction_result = response.json()
    validate_prediction_response(prediction_result)

    decision_support_url = base_url + "/decision_support"
    response = requests.post(decision_support_url, json=SAMPLE_PAYLOAD, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(
            f"FAIL: {decision_support_url} returned {response.status_code}: {response.text[:500]}"
        )
    validate_decision_support_response(response.json())
    print_summary(prediction_result, mode=f"http:{prediction_url}")


def run_in_process() -> None:
    from fastapi.testclient import TestClient
    from backend.app import app

    client = TestClient(app)
    response = client.get("/health")
    if response.status_code != 200:
        raise RuntimeError(f"FAIL: in-process /health returned {response.status_code}: {response.text[:500]}")
    if response.json().get("status") != "ok":
        raise RuntimeError("FAIL: in-process /health did not return status=ok")

    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    if response.status_code != 200:
        raise RuntimeError(f"FAIL: in-process /predict returned {response.status_code}: {response.text[:500]}")
    prediction_result = response.json()
    validate_prediction_response(prediction_result)

    response = client.post("/decision_support", json=SAMPLE_PAYLOAD)
    if response.status_code != 200:
        raise RuntimeError(
            f"FAIL: in-process /decision_support returned {response.status_code}: {response.text[:500]}"
        )
    validate_decision_support_response(response.json())
    print_summary(prediction_result, mode="in-process:/predict")


def main() -> int:
    api_base_url = os.environ.get("PROJECT2_API_URL", "").strip()
    try:
        if api_base_url:
            run_against_http(api_base_url)
        else:
            run_in_process()
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
