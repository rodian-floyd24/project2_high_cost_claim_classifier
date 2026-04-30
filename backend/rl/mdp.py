from __future__ import annotations

from dataclasses import dataclass

from .config import (
    ACTIONS,
    ACTION_TABLE,
    CHRONIC_BURDEN_BINS,
    CHRONIC_BURDENS,
    HIGH_COST_PROBABILITY_THRESHOLD,
    PRIOR_INTERVENTIONS,
    PRIOR_INTERVENTION_EFFECT,
    PRIOR_INTERVENTION_STATES,
    REWARD_CONFIG,
    RISK_ORDER,
    RISK_TIERS,
    RISK_TIER_ACTION_EFFECT,
    RISK_TIER_BINS,
    UTILIZATION_BIN_RULES,
    UTILIZATION_LEVELS,
)


@dataclass(frozen=True)
class StateComponents:
    risk_tier: str
    chronic_burden: str
    utilization_intensity: str
    prior_intervention_status: str


def compute_utilization_intensity(total_claim_count: int, total_claim_days: int) -> str:
    if total_claim_count <= 8 and total_claim_days <= 5:
        return "low"
    if total_claim_count <= 20 and total_claim_days <= 15:
        return "medium"
    return "high"


def total_claim_count_from_profile(profile) -> int:
    return (
        int(profile.inpatient_claim_count)
        + int(profile.outpatient_claim_count)
        + int(profile.carrier_claim_count)
        + int(profile.pde_claim_count)
    )


def assign_risk_tier(risk_probability: float) -> str:
    for label, lower, upper in RISK_TIER_BINS:
        if lower <= risk_probability < upper:
            return label
    return "very_high"


def assign_chronic_burden(chronic_condition_count: int) -> str:
    for label, lower, upper in CHRONIC_BURDEN_BINS:
        if lower <= chronic_condition_count <= upper:
            return label
    return "high"


def assign_utilization_intensity(
    inpatient_claim_count: int,
    outpatient_claim_count: int,
    carrier_claim_count: int,
    pde_claim_count: int,
    total_claim_days: int,
) -> str:
    total_claim_count = (
        inpatient_claim_count
        + outpatient_claim_count
        + carrier_claim_count
        + pde_claim_count
    )
    return compute_utilization_intensity(total_claim_count, total_claim_days)


def state_to_id(risk_tier: str, chronic_burden: str, utilization_intensity: str, prior_intervention_status: str) -> int:
    r = RISK_TIERS.index(risk_tier)
    c = CHRONIC_BURDENS.index(chronic_burden)
    u = UTILIZATION_LEVELS.index(utilization_intensity)
    i = PRIOR_INTERVENTIONS.index(prior_intervention_status)
    return (((r * len(CHRONIC_BURDENS)) + c) * len(UTILIZATION_LEVELS) + u) * len(PRIOR_INTERVENTIONS) + i


def id_to_state(state_id: int) -> tuple[str, str, str, str]:
    i = state_id % len(PRIOR_INTERVENTIONS)
    state_id //= len(PRIOR_INTERVENTIONS)
    u = state_id % len(UTILIZATION_LEVELS)
    state_id //= len(UTILIZATION_LEVELS)
    c = state_id % len(CHRONIC_BURDENS)
    state_id //= len(CHRONIC_BURDENS)
    r = state_id
    return (
        RISK_TIERS[r],
        CHRONIC_BURDENS[c],
        UTILIZATION_LEVELS[u],
        PRIOR_INTERVENTIONS[i],
    )


def build_mdp_state(
    risk_probability: float,
    chronic_condition_count: int,
    inpatient_claim_count: int,
    outpatient_claim_count: int,
    carrier_claim_count: int,
    pde_claim_count: int,
    total_claim_days: int,
    prior_intervention_status: str,
) -> dict[str, object]:
    risk_tier = assign_risk_tier(risk_probability)
    chronic_burden = assign_chronic_burden(chronic_condition_count)
    utilization_intensity = assign_utilization_intensity(
        inpatient_claim_count,
        outpatient_claim_count,
        carrier_claim_count,
        pde_claim_count,
        total_claim_days,
    )
    state_id = state_to_id(risk_tier, chronic_burden, utilization_intensity, prior_intervention_status)
    return {
        "risk_tier": risk_tier,
        "chronic_burden": chronic_burden,
        "utilization_intensity": utilization_intensity,
        "prior_intervention_status": prior_intervention_status,
        "state_id": state_id,
    }


def state_components_from_profile(profile, probability: float) -> StateComponents:
    return StateComponents(
        risk_tier=assign_risk_tier(probability),
        chronic_burden=assign_chronic_burden(int(profile.chronic_condition_count)),
        utilization_intensity=assign_utilization_intensity(
            int(profile.inpatient_claim_count),
            int(profile.outpatient_claim_count),
            int(profile.carrier_claim_count),
            int(profile.pde_claim_count),
            int(profile.total_claim_days),
        ),
        prior_intervention_status=str(profile.prior_intervention_status),
    )


def state_id_from_components(components: StateComponents) -> int:
    return state_to_id(
        components.risk_tier,
        components.chronic_burden,
        components.utilization_intensity,
        components.prior_intervention_status,
    )


def state_label(components: StateComponents) -> str:
    prior_label = {
        "none": "no recent intervention",
        "recent_low_touch": "recent low-touch outreach",
        "recent_intensive": "recent intensive intervention",
    }[components.prior_intervention_status]
    return (
        f"{components.risk_tier.replace('_', ' ').capitalize()} risk, "
        f"{components.chronic_burden} chronic burden, "
        f"{components.utilization_intensity} utilization, "
        f"{prior_label}"
    )


def adjusted_next_period_risk(
    base_risk_probability: float,
    action_name: str,
    risk_tier: str,
    prior_intervention_status: str,
    chronic_burden: str | None = None,
    utilization_intensity: str | None = None,
) -> float:
    action_multiplier = ACTION_TABLE[action_name]["baseline_risk_multiplier"]
    raw_reduction = 1.0 - action_multiplier
    gain = RISK_TIER_ACTION_EFFECT[risk_tier]
    prior_factor = PRIOR_INTERVENTION_EFFECT[prior_intervention_status]
    suitability_factor = 1.0
    if action_name == "low_touch_outreach":
        if risk_tier == "low":
            suitability_factor = 0.45
        elif risk_tier in {"elevated", "high"}:
            suitability_factor = 1.05
    elif action_name == "care_coordination_call":
        if risk_tier == "low" and chronic_burden == "low":
            suitability_factor = 0.15
        elif utilization_intensity == "high" or chronic_burden in {"medium", "high"}:
            suitability_factor = 1.15
    elif action_name == "intensive_case_management":
        if risk_tier == "low":
            suitability_factor = 0.20
        elif risk_tier == "elevated":
            suitability_factor = 0.55
        elif risk_tier == "high":
            suitability_factor = 1.10
        elif risk_tier == "very_high":
            suitability_factor = 1.35
        if chronic_burden == "high":
            suitability_factor *= 1.10
        if utilization_intensity == "high":
            suitability_factor *= 1.12
    effective_reduction = raw_reduction * gain * prior_factor * suitability_factor
    effective_reduction = min(0.85, max(0.0, effective_reduction))
    next_risk = base_risk_probability * (1.0 - effective_reduction)
    return max(0.01, min(0.99, next_risk))


def next_risk_tier_from_probability(probability: float) -> str:
    return assign_risk_tier(probability)


def compute_reward(
    action_name: str,
    current_risk_tier: str,
    next_risk_tier: str,
    became_high_cost_next_period: bool,
) -> float:
    action_cost = ACTION_TABLE[action_name]["intervention_cost"]
    reward = -action_cost
    if became_high_cost_next_period:
        reward += REWARD_CONFIG["high_cost_penalty"]
    else:
        reward += REWARD_CONFIG["non_high_cost_bonus"]
    if RISK_ORDER[next_risk_tier] < RISK_ORDER[current_risk_tier]:
        reward += REWARD_CONFIG["risk_improvement_bonus"]
    elif RISK_ORDER[next_risk_tier] > RISK_ORDER[current_risk_tier]:
        reward += REWARD_CONFIG["risk_worsening_penalty"]
    overtreatment = current_risk_tier == "low" and action_name == "intensive_case_management"
    if overtreatment:
        reward += REWARD_CONFIG["overtreatment_penalty"]
    return reward


def next_prior_intervention_status(action_name: str) -> str:
    if action_name == "no_action":
        return "none"
    if action_name == "low_touch_outreach":
        return "recent_low_touch"
    return "recent_intensive"


def simulate_step(probability: float, components: StateComponents, action_name: str, rng) -> tuple[int, float, StateComponents]:
    # In v1 the episode keeps chronic burden fixed and only evolves risk tier,
    # utilization intensity, and prior intervention status.
    next_probability = adjusted_next_period_risk(
        base_risk_probability=probability,
        action_name=action_name,
        risk_tier=components.risk_tier,
        prior_intervention_status=components.prior_intervention_status,
        chronic_burden=components.chronic_burden,
        utilization_intensity=components.utilization_intensity,
    )
    became_high_cost = bool(rng.random() < next_probability)
    next_risk_tier = next_risk_tier_from_probability(next_probability)

    current_utilization_index = UTILIZATION_LEVELS.index(components.utilization_intensity)
    next_utilization_index = current_utilization_index
    utilization_improvement_probability = 0.0
    if action_name == "low_touch_outreach":
        utilization_improvement_probability = 0.12
    elif action_name == "care_coordination_call":
        utilization_improvement_probability = 0.24
    elif action_name == "intensive_case_management":
        utilization_improvement_probability = 0.30

    if components.utilization_intensity == "high":
        utilization_improvement_probability += 0.14
    elif components.utilization_intensity == "medium":
        utilization_improvement_probability += 0.06

    if components.risk_tier == "very_high":
        utilization_improvement_probability += 0.10
    elif components.risk_tier == "high":
        utilization_improvement_probability += 0.06

    if components.chronic_burden == "high":
        utilization_improvement_probability += 0.08

    if action_name == "care_coordination_call" and components.risk_tier == "low" and components.chronic_burden == "low":
        utilization_improvement_probability *= 0.10
    if action_name == "low_touch_outreach" and components.risk_tier == "low":
        utilization_improvement_probability *= 0.30
    if action_name == "intensive_case_management" and components.risk_tier in {"low", "elevated"}:
        utilization_improvement_probability *= 0.35

    if current_utilization_index > 0 and not became_high_cost and rng.random() < min(0.75, utilization_improvement_probability):
        next_utilization_index -= 1
    elif became_high_cost and current_utilization_index < len(UTILIZATION_LEVELS) - 1:
        if rng.random() < 0.45:
            next_utilization_index += 1

    next_components = StateComponents(
        risk_tier=next_risk_tier,
        chronic_burden=components.chronic_burden,
        utilization_intensity=UTILIZATION_LEVELS[next_utilization_index],
        prior_intervention_status=next_prior_intervention_status(action_name),
    )
    reward = compute_reward(
        action_name=action_name,
        current_risk_tier=components.risk_tier,
        next_risk_tier=next_components.risk_tier,
        became_high_cost_next_period=became_high_cost,
    )
    return state_id_from_components(next_components), reward, next_components


def action_catalog() -> list[dict[str, object]]:
    return [{"key": action_name, **ACTION_TABLE[action_name]} for action_name in ACTIONS]


def action_key_from_index(index: int) -> str:
    return ACTIONS[index]


def build_policy_rationale(state: dict[str, object], recommended_action: str) -> str:
    rt = str(state["risk_tier"])
    cb = str(state["chronic_burden"])
    ui = str(state["utilization_intensity"])
    if recommended_action == "no_action":
        return (
            f"The current state is {rt} risk with {cb} chronic burden and {ui} utilization. "
            f"Given the current policy assumptions, intervention cost outweighs expected incremental benefit."
        )
    if recommended_action == "low_touch_outreach":
        return (
            f"The beneficiary is {rt} risk with {cb} chronic burden and {ui} utilization. "
            f"Low-touch outreach offers a favorable low-cost intervention under the current policy."
        )
    if recommended_action == "care_coordination_call":
        return (
            f"The beneficiary is {rt} risk with {cb} chronic burden and {ui} utilization. "
            f"Care coordination provides the best balance of intervention cost and expected downstream risk reduction."
        )
    return (
        f"The beneficiary is {rt} risk with {cb} chronic burden and {ui} utilization. "
        f"Under the current simulated policy assumptions, intensive case management has the highest expected long-run value."
    )


def state_snapshot(probability: float, components: StateComponents) -> dict[str, object]:
    return {
        "risk_tier": components.risk_tier,
        "chronic_burden": components.chronic_burden,
        "utilization_intensity": components.utilization_intensity,
        "prior_intervention_status": components.prior_intervention_status,
        "state_id": state_id_from_components(components),
        "label": state_label(components),
        "baseline_risk_probability": round(probability, 6),
        "high_cost_threshold": HIGH_COST_PROBABILITY_THRESHOLD,
    }
