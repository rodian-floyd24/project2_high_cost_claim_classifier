from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import ACTIONS, CHRONIC_BURDENS, PRIOR_INTERVENTIONS, Q_LEARNING_CONFIG, RISK_TIERS, STATE_COUNT, UTILIZATION_LEVELS, rl_metadata
from .mdp import StateComponents, action_key_from_index, simulate_step, state_id_from_components


def epsilon_for_episode(episode: int) -> float:
    start = Q_LEARNING_CONFIG.epsilon_start
    end = Q_LEARNING_CONFIG.epsilon_end
    total = max(1, Q_LEARNING_CONFIG.episodes - 1)
    fraction = episode / total
    return max(end, start - (start - end) * fraction)


def choose_action(q_table: np.ndarray, state_id: int, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(len(ACTIONS)))
    return int(np.argmax(q_table[state_id]))


def random_initial_state(rng: np.random.Generator) -> tuple[float, StateComponents]:
    # These tier-level probabilities are only a bootstrap for offline RL training.
    # Online policy use starts from the actual supervised-model probability.
    risk_tier = RISK_TIERS[int(rng.integers(len(RISK_TIERS)))]
    chronic_burden = CHRONIC_BURDENS[int(rng.integers(len(CHRONIC_BURDENS)))]
    utilization_intensity = UTILIZATION_LEVELS[int(rng.integers(len(UTILIZATION_LEVELS)))]
    prior_intervention_status = PRIOR_INTERVENTIONS[int(rng.integers(len(PRIOR_INTERVENTIONS)))]
    base_probability_by_tier = {
        "low": 0.10,
        "elevated": 0.30,
        "high": 0.50,
        "very_high": 0.72,
    }
    probability = base_probability_by_tier[risk_tier]
    if chronic_burden == "medium":
        probability = min(0.95, probability + 0.02)
    if chronic_burden == "high":
        probability = min(0.95, probability + 0.05)
    if utilization_intensity == "medium":
        probability = min(0.95, probability + 0.03)
    if utilization_intensity == "high":
        probability = min(0.95, probability + 0.10)
    return probability, StateComponents(
        risk_tier=risk_tier,
        chronic_burden=chronic_burden,
        utilization_intensity=utilization_intensity,
        prior_intervention_status=prior_intervention_status,
    )


def probability_from_state(components: StateComponents) -> float:
    # This helper supports offline training rollouts only. It is not used to replace
    # the empirical supervised-model probability during online recommendation.
    base_probability_by_tier = {
        "low": 0.10,
        "elevated": 0.30,
        "high": 0.50,
        "very_high": 0.72,
    }
    probability = base_probability_by_tier[components.risk_tier]
    if components.chronic_burden == "medium":
        probability = min(0.95, probability + 0.02)
    if components.chronic_burden == "high":
        probability = min(0.95, probability + 0.05)
    if components.utilization_intensity == "medium":
        probability = min(0.95, probability + 0.03)
    if components.utilization_intensity == "high":
        probability = min(0.95, probability + 0.10)
    return probability


def train_q_table() -> np.ndarray:
    q_table = np.zeros((STATE_COUNT, len(ACTIONS)), dtype=float)
    rng = np.random.default_rng(Q_LEARNING_CONFIG.random_seed)
    for episode in range(Q_LEARNING_CONFIG.episodes):
        epsilon = epsilon_for_episode(episode)
        probability, components = random_initial_state(rng)
        state_id = state_id_from_components(components)
        for _ in range(Q_LEARNING_CONFIG.max_steps_per_episode):
            action_index = choose_action(q_table, state_id, epsilon, rng)
            action_name = action_key_from_index(action_index)
            next_state_id, reward, next_components = simulate_step(probability, components, action_name, rng)
            td_target = reward + Q_LEARNING_CONFIG.gamma * float(np.max(q_table[next_state_id]))
            td_error = td_target - q_table[state_id, action_index]
            q_table[state_id, action_index] += Q_LEARNING_CONFIG.alpha * td_error
            state_id = next_state_id
            components = next_components
            probability = probability_from_state(components)
    return q_table


def save_q_table(path: Path, q_table: np.ndarray) -> None:
    rows = {str(index): [round(float(value), 6) for value in row] for index, row in enumerate(q_table)}
    path.write_text(json.dumps(rows, indent=2))


def load_q_table(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    table = np.zeros((STATE_COUNT, len(ACTIONS)), dtype=float)
    for state_id, row in data.items():
        table[int(state_id)] = np.array(row, dtype=float)
    return table


def save_metadata(path: Path) -> None:
    path.write_text(json.dumps(rl_metadata(), indent=2))
