"""
Test Script for Actuarial Decision-Support Prototype
====================================================
Sends a sample beneficiary profile to the prediction endpoint and prints the result.

Default behavior:
- If PROJECT2_API_URL is set, sends an HTTP request to that deployed API.
- Otherwise, runs the FastAPI app in-process with TestClient and validates the
  same endpoint locally without requiring a running server.

Requirements:
- Local fallback: `pip install -r requirements-dev.txt`
- Deployed API mode: `pip install -r requirements-dev.txt`

Usage:
- `python test_project.py`
- `PROJECT2_API_URL=https://your-app-url.com python test_project.py`
"""

from __future__ import annotations

import json
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


def print_summary(result: dict[str, Any], mode: str) -> None:
    prediction = result["prediction"] if "prediction" in result else result

    print(f"Mode: {mode}")
    print(f"Risk probability: {prediction['risk_probability']:.4f}")
    print(f"Risk tier: {prediction['risk_tier']}")
    print(f"Predicted high cost: {prediction['predicted_high_cost']}")
    print(f"Recommended action: {prediction['recommended_action']}")
    print("PASS")


def validate_prediction_response(result: dict[str, Any]) -> None:
    required_prediction_keys = {
        "risk_probability",
        "risk_tier",
        "predicted_high_cost",
        "recommended_action",
        "decision_threshold",
    }
    missing = required_prediction_keys - set(result)
    if missing:
        raise RuntimeError(f"Missing prediction keys: {sorted(missing)}")

    if not 0.0 <= float(result["risk_probability"]) <= 1.0:
        raise RuntimeError("risk_probability outside [0, 1]")
    if not result["recommended_action"]:
        raise RuntimeError("recommended_action missing")


def validate_decision_support_response(result: dict[str, Any]) -> None:
    required_top_level = {"prediction", "state", "recommendation", "simulation", "disclaimer"}
    missing = required_top_level - set(result)
    if missing:
        raise RuntimeError(f"Missing top-level keys: {sorted(missing)}")

    prediction = result["prediction"]
    recommendation = result["recommendation"]
    state = result["state"]["current_state"]

    if not 0.0 <= float(prediction["risk_probability"]) <= 1.0:
        raise RuntimeError("risk_probability outside [0, 1]")
    if not recommendation["recommended_action"]:
        raise RuntimeError("recommended_action missing")
    if "label" not in state:
        raise RuntimeError("state label missing")


def run_against_http(api_base_url: str) -> None:
    import requests

    base_url = api_base_url.rstrip("/")
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
