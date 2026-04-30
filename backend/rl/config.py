from __future__ import annotations

from dataclasses import asdict, dataclass


RISK_TIERS = ["low", "elevated", "high", "very_high"]
CHRONIC_BURDENS = ["low", "medium", "high"]
UTILIZATION_LEVELS = ["low", "medium", "high"]
PRIOR_INTERVENTIONS = ["none", "recent_low_touch", "recent_intensive"]

RISK_TIER_BINS = [
    ("low", 0.00, 0.20),
    ("elevated", 0.20, 0.40),
    ("high", 0.40, 0.60),
    ("very_high", 0.60, 1.01),
]

CHRONIC_BURDEN_BINS = [
    ("low", 0, 2),
    ("medium", 3, 5),
    ("high", 6, 99),
]

UTILIZATION_BIN_RULES = {
    "low": {"max_total_claim_count": 8, "max_total_claim_days": 5},
    "medium": {"max_total_claim_count": 20, "max_total_claim_days": 15},
    "high": {"max_total_claim_count": None, "max_total_claim_days": None},
}

PRIOR_INTERVENTION_STATES = [
    "none",
    "recent_low_touch",
    "recent_intensive",
]

ACTIONS = [
    "no_action",
    "low_touch_outreach",
    "care_coordination_call",
    "intensive_case_management",
]

ACTION_TABLE = {
    "no_action": {
        "action_id": 0,
        "display_name": "No action",
        "description": "Observe only; no intervention cost incurred.",
        "intervention_cost": 0.0,
        "baseline_risk_multiplier": 1.00,
    },
    "low_touch_outreach": {
        "action_id": 1,
        "display_name": "Low-touch outreach",
        "description": "Digital reminder, education, or light outreach.",
        "intervention_cost": 1.0,
        "baseline_risk_multiplier": 0.95,
    },
    "care_coordination_call": {
        "action_id": 2,
        "display_name": "Care coordination call",
        "description": "Moderate-cost nurse or care coordination call.",
        "intervention_cost": 3.0,
        "baseline_risk_multiplier": 0.88,
    },
    "intensive_case_management": {
        "action_id": 3,
        "display_name": "Intensive case management",
        "description": "High-touch intervention with highest operational cost.",
        "intervention_cost": 6.0,
        "baseline_risk_multiplier": 0.80,
    },
}

REWARD_CONFIG = {
    "high_cost_penalty": -12.0,
    "non_high_cost_bonus": 3.0,
    "risk_improvement_bonus": 2.0,
    "risk_worsening_penalty": -2.0,
    "overtreatment_penalty": -2.0,
}

RISK_TIER_ACTION_EFFECT = {
    "low": 1.00,
    "elevated": 1.05,
    "high": 1.10,
    "very_high": 1.15,
}

PRIOR_INTERVENTION_EFFECT = {
    "none": 1.00,
    "recent_low_touch": 0.95,
    "recent_intensive": 0.85,
}

HIGH_COST_PROBABILITY_THRESHOLD = 0.50
RISK_ORDER = {risk_tier: idx for idx, risk_tier in enumerate(RISK_TIERS)}
STATE_COUNT = len(RISK_TIERS) * len(CHRONIC_BURDENS) * len(UTILIZATION_LEVELS) * len(PRIOR_INTERVENTIONS)


@dataclass(frozen=True)
class QLearningHyperparameters:
    alpha: float = 0.10
    gamma: float = 0.95
    epsilon_start: float = 0.20
    epsilon_end: float = 0.02
    episodes: int = 50000
    max_steps_per_episode: int = 12
    random_seed: int = 42


Q_LEARNING_CONFIG = QLearningHyperparameters()

SIMULATION_DISCLAIMER = (
    "The policy layer is a simulated MDP/Q-learning intervention prototype built on stylized "
    "transition and reward assumptions. It is not causally learned from real intervention-response histories."
)


def rl_metadata() -> dict[str, object]:
    return {
        "state_space": {
            "risk_tiers": RISK_TIERS,
            "risk_tier_bins": RISK_TIER_BINS,
            "chronic_burdens": CHRONIC_BURDENS,
            "chronic_burden_bins": CHRONIC_BURDEN_BINS,
            "utilization_levels": UTILIZATION_LEVELS,
            "utilization_bin_rules": UTILIZATION_BIN_RULES,
            "prior_interventions": PRIOR_INTERVENTIONS,
            "state_count": STATE_COUNT,
        },
        "actions": ACTION_TABLE,
        "reward_config": REWARD_CONFIG,
        "risk_tier_action_effect": RISK_TIER_ACTION_EFFECT,
        "prior_intervention_effect": PRIOR_INTERVENTION_EFFECT,
        "q_learning": asdict(Q_LEARNING_CONFIG),
        "training_notes": {
            "offline_initialization": (
                "Offline Q-learning training bootstraps episodes from stylized initial risk probabilities "
                "by risk tier. Those bootstrap probabilities are used only to initialize simulated training episodes."
            ),
            "online_policy_use": (
                "Online policy use starts from the actual supervised-model probability for the beneficiary and "
                "maps that empirical score into the MDP state and transition logic."
            ),
            "episode_dynamics": (
                "Chronic burden is treated as fixed within the simulated episode horizon in v1, while risk tier, "
                "utilization intensity, and prior intervention status can change across steps."
            ),
        },
        "disclaimer": SIMULATION_DISCLAIMER,
    }
