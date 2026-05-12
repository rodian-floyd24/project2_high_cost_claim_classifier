from __future__ import annotations

import json
from pathlib import Path

from shared.feature_contract import FEATURE_VERSION

MODEL_NAME = "high_cost_claim_classifier"
MODEL_VERSION = "actuarial_decision_support_v1"
MODEL_PYTHON_VERSION = "3.11.10"
MODEL_SKLEARN_VERSION = "1.3.0"
SPLIT_VERSION = "xxhash64_bene_id_mod_100_v2_beneficiary_hash_holdout"
TARGET_DEFINITION = "next_year_top_decile_training_threshold"
MODEL_METADATA_PATH = Path(__file__).resolve().parent / "model_artifacts" / "model_metadata.json"


def load_model_metadata() -> dict[str, object]:
    if MODEL_METADATA_PATH.exists():
        return json.loads(MODEL_METADATA_PATH.read_text())
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "python_version": MODEL_PYTHON_VERSION,
        "sklearn_version": MODEL_SKLEARN_VERSION,
        "feature_version": FEATURE_VERSION,
        "split_version": SPLIT_VERSION,
        "target_definition": TARGET_DEFINITION,
    }


def score_to_operating_tier(risk_score: float) -> str:
    score = max(0.0, min(float(risk_score), 1.0))
    if score >= 0.30:
        return "very_high"
    if score >= 0.20:
        return "high"
    if score >= 0.10:
        return "elevated"
    return "routine"


def operating_policy_for_score(risk_score: float) -> tuple[str, float, str]:
    score = max(0.0, min(float(risk_score), 1.0))
    tier = score_to_operating_tier(score)
    actions = {
        "very_high": "intensive_case_management_review",
        "high": "moderate_outreach",
        "elevated": "monitor_and_flag_for_next_review",
        "routine": "routine_monitoring",
    }
    return tier, score, actions[tier]


def deterministic_feature_order(model) -> list[str]:
    signature = getattr(model, "metadata", None)
    if signature is None:
        return []
    model_signature = getattr(signature, "signature", None)
    if model_signature is None or model_signature.inputs is None:
        return []
    return [item.name for item in model_signature.inputs.inputs]
