from __future__ import annotations

import json
import math
from pathlib import Path

from shared.feature_contract import FEATURE_CONTRACT_VERSION, FEATURE_VERSION

MODEL_NAME = "gradient_boosting"
MODEL_VERSION = "actuarial_decision_support_v2"
MODEL_PYTHON_VERSION = "3.11.10"
MODEL_SKLEARN_VERSION = "1.3.0"
SPLIT_VERSION = "xxhash64_bene_id_mod_100_v2_beneficiary_hash_holdout"
TARGET_DEFINITION = "next_year_top_decile_training_threshold"
CALIBRATION_METHOD = "isotonic"
CALIBRATION_STATUS = "calibrated"
PROBABILITY_INTERPRETATION = "calibrated_probability_for_risk_tiering_not_a_definitive_prediction_of_individual_claims"
RANKING_USE_STATEMENT = "rank_order_risk_usefulness_is_prioritized_for_targeted_operations_over_strict_classification_thresholds"
MODEL_METADATA_PATH = Path(__file__).resolve().parent / "model_artifacts" / "model_metadata.json"
REFERENCE_DISTRIBUTION_PATH = Path(__file__).resolve().parent / "model_artifacts" / "reference_distribution.json"


def load_model_metadata() -> dict[str, object]:
    if MODEL_METADATA_PATH.exists():
        return json.loads(MODEL_METADATA_PATH.read_text())
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "python_version": MODEL_PYTHON_VERSION,
        "sklearn_version": MODEL_SKLEARN_VERSION,
        "feature_version": FEATURE_VERSION,
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "split_version": SPLIT_VERSION,
        "target_definition": TARGET_DEFINITION,
        "calibration_method": CALIBRATION_METHOD,
        "calibration_status": CALIBRATION_STATUS,
        "probability_interpretation": PROBABILITY_INTERPRETATION,
        "ranking_use_statement": RANKING_USE_STATEMENT,
    }


def load_reference_distribution() -> list[float]:
    if not REFERENCE_DISTRIBUTION_PATH.exists():
        return []
    payload = json.loads(REFERENCE_DISTRIBUTION_PATH.read_text())
    values = payload.get("predicted_probabilities", payload) if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        raise ValueError("reference_distribution.json must be a list or contain predicted_probabilities")
    return sorted(max(0.0, min(float(value), 1.0)) for value in values)


def compute_percentile_score(probability: float, reference_distribution: list[float]) -> int:
    if not reference_distribution:
        raise ValueError("reference_distribution cannot be empty")
    probability = max(0.0, min(float(probability), 1.0))
    count_less_equal = sum(p <= probability for p in reference_distribution)
    percentile = count_less_equal / len(reference_distribution)
    return int(round(max(0.0, min(100.0, 100.0 * percentile))))


def monotone_probability_score(p: float, scale: float = 0.10) -> int:
    """
    Maps calibrated probability to a 0-100 display score.
    A probability near 10% maps near the upper range because this is
    a rare-event high-cost classifier.
    """
    p = max(0.0, min(float(p), 1.0))
    score = 100.0 * (1.0 - math.exp(-p / scale))
    return int(round(max(0.0, min(score, 100.0))))


def compute_risk_score(calibrated_probability: float, reference_distribution: list[float] | None = None) -> tuple[int, str]:
    if reference_distribution:
        score = compute_percentile_score(calibrated_probability, reference_distribution)
        return score, "percentile_rank_calibrated_probability"
    score = monotone_probability_score(calibrated_probability)
    return score, "monotone_probability_fallback_v1"


def assign_risk_tier(score: int) -> str:
    if score >= 95:
        return "very_high"
    if score >= 90:
        return "high"
    if score >= 75:
        return "elevated"
    return "low"


def generate_reason_codes(row: dict[str, float | int | str], thresholds: dict[str, float]) -> list[str]:
    reasons = []

    if int(row.get("chronic_condition_count", 0)) >= 6:
        reasons.append("High chronic condition burden")

    if float(row.get("claims_per_enrollment_month", 0.0)) >= thresholds.get("claims_per_enrollment_month", float("inf")):
        reasons.append("High claims per enrollment month")

    if float(row.get("provider_fragmentation_index", 0.0)) >= thresholds.get("provider_fragmentation_index", float("inf")):
        reasons.append("High provider fragmentation")

    if int(row.get("any_carrier_claim", 0)) == 1:
        reasons.append("Carrier claim activity present")

    if float(row.get("total_claim_days_log1p", 0.0)) >= thresholds.get("total_claim_days_log1p", float("inf")):
        reasons.append("High claim-day utilization")

    return reasons[:3]


def deterministic_feature_order(model) -> list[str]:
    signature = getattr(model, "metadata", None)
    if signature is None:
        return []
    model_signature = getattr(signature, "signature", None)
    if model_signature is None or model_signature.inputs is None:
        return []
    return [item.name for item in model_signature.inputs.inputs]
