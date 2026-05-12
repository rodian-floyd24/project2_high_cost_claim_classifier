from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from shared.feature_contract import MODEL_FEATURE_ORDER, MODEL_INT32_FIELDS, MODEL_INT64_FIELDS

from ..scoring import deterministic_feature_order
from .config import ACTION_TABLE, SIMULATION_DISCLAIMER
from .mdp import (
    action_catalog,
    adjusted_next_period_risk,
    build_policy_rationale,
    compute_reward,
    next_risk_tier_from_probability,
    state_components_from_profile,
    state_id_from_components,
    state_snapshot,
)


def expected_model_feature_order(model) -> list[str]:
    signature_order = deterministic_feature_order(model)
    if signature_order:
        return signature_order
    return MODEL_FEATURE_ORDER


def score_probability(feature_row: dict[str, Any], model) -> float:
    features = pd.DataFrame([feature_row])
    for column in MODEL_INT32_FIELDS:
        if column in features.columns:
            features[column] = features[column].astype(np.int32)
    for column in MODEL_INT64_FIELDS:
        if column in features.columns:
            features[column] = features[column].astype(np.int64)
    feature_order = expected_model_feature_order(model)
    missing_columns = [column for column in feature_order if column not in features.columns]
    if missing_columns:
        raise ValueError(f"Missing engineered model features: {', '.join(missing_columns)}")
    return float(model.predict_proba(features[feature_order])[0][1])


def profile_to_state(feature_row: dict[str, Any], profile, model) -> tuple[float, dict[str, object]]:
    probability = score_probability(feature_row, model)
    components = state_components_from_profile(profile, probability)
    snapshot = state_snapshot(probability, components)
    return probability, snapshot


def action_values_for_profile(feature_row: dict[str, Any], profile, model, q_table: np.ndarray) -> dict[str, object]:
    probability = score_probability(feature_row, model)
    components = state_components_from_profile(profile, probability)
    state_id = state_id_from_components(components)
    q_values = q_table[state_id]

    action_values = []
    q_values_map: dict[str, float] = {}
    for action, q_value in zip(action_catalog(), q_values):
        next_probability = adjusted_next_period_risk(
            base_risk_probability=probability,
            action_name=action["key"],
            risk_tier=components.risk_tier,
            prior_intervention_status=components.prior_intervention_status,
            chronic_burden=components.chronic_burden,
            utilization_intensity=components.utilization_intensity,
        )
        next_risk_tier = next_risk_tier_from_probability(next_probability)
        became_high_cost = next_probability >= 0.50
        immediate_reward = compute_reward(
            action_name=action["key"],
            current_risk_tier=components.risk_tier,
            next_risk_tier=next_risk_tier,
            became_high_cost_next_period=became_high_cost,
        )
        q_value_float = round(float(q_value), 6)
        q_values_map[action["key"]] = q_value_float
        action_values.append(
            {
                "action": action["key"],
                "action_label": action["display_name"],
                "q_value": q_value_float,
                "expected_next_risk_probability": round(float(next_probability), 6),
                "expected_immediate_reward": round(float(immediate_reward), 6),
            }
        )

    action_values.sort(key=lambda item: item["q_value"], reverse=True)
    recommended = action_values[0]
    state = state_snapshot(probability, components)
    explanation = build_policy_rationale(state, recommended["action"])
    return {
        "risk_probability": probability,
        "risk_tier": components.risk_tier,
        "state": state,
        "recommended_action": recommended["action"],
        "recommended_action_label": recommended["action_label"],
        "recommended_action_display": recommended["action_label"],
        "expected_long_run_value": recommended["q_value"],
        "q_values": q_values_map,
        "action_values": action_values,
        "policy_explanation": explanation,
        "disclaimer": SIMULATION_DISCLAIMER,
    }


def compare_all_actions(feature_row: dict[str, Any], profile, model, q_table: np.ndarray) -> dict[str, object]:
    recommendation = action_values_for_profile(feature_row, profile, model, q_table)
    state = recommendation["state"]
    comparisons = []
    for item in recommendation["action_values"]:
        risk_delta = round(float(item["expected_next_risk_probability"] - recommendation["risk_probability"]), 6)
        comparisons.append(
            {
                **item,
                "expected_risk_delta": risk_delta,
                "is_recommended": item["action"] == recommendation["recommended_action"],
            }
        )
    return {
        "state": state,
        "baseline_risk_probability": round(float(recommendation["risk_probability"]), 6),
        "risk_tier": recommendation["risk_tier"],
        "recommended_action": recommendation["recommended_action"],
        "recommended_action_label": recommendation["recommended_action_label"],
        "recommended_action_display": recommendation["recommended_action_display"],
        "q_values": recommendation["q_values"],
        "comparisons": comparisons,
        "policy_explanation": recommendation["policy_explanation"],
        "disclaimer": recommendation["disclaimer"],
    }
