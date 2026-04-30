from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_auc(y_true, y_prob, curve: str = "roc") -> float:
    labels = np.asarray(y_true)
    if len(np.unique(labels)) < 2:
        return math.nan
    if curve == "pr":
        return float(average_precision_score(y_true, y_prob))
    return float(roc_auc_score(y_true, y_prob))


def classification_metrics(y_true, y_prob, threshold: float) -> dict[str, float | int]:
    labels = np.asarray(y_true).astype(int)
    probabilities = np.asarray(y_prob).astype(float)
    predictions = (probabilities >= float(threshold)).astype(int)
    accuracy = float(accuracy_score(labels, predictions))
    return {
        "row_count": int(len(labels)),
        "positive_rate": float(labels.mean()) if len(labels) else 0.0,
        "accuracy": accuracy,
        "test_misclassification_error": float(1.0 - accuracy),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": safe_auc(labels, probabilities, "roc"),
        "pr_auc": safe_auc(labels, probabilities, "pr"),
        "brier_score": float(brier_score_loss(labels, probabilities)) if len(labels) else math.nan,
        "decision_threshold": float(threshold),
    }


def topk_metrics(y_true, y_prob, selected_fractions: Iterable[float]) -> list[dict[str, float | int]]:
    ranked = pd.DataFrame({"label": np.asarray(y_true).astype(int), "score": np.asarray(y_prob).astype(float)})
    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
    total_rows = len(ranked)
    total_positives = float(ranked["label"].sum())
    base_rate = float(ranked["label"].mean()) if total_rows else 0.0
    rows: list[dict[str, float | int]] = []

    for fraction in selected_fractions:
        selected_count = max(1, int(math.ceil(total_rows * float(fraction)))) if total_rows else 0
        selected = ranked.head(selected_count)
        captured_positive_count = int(selected["label"].sum()) if selected_count else 0
        observed_positive_rate = float(selected["label"].mean()) if selected_count else 0.0
        rows.append(
            {
                "selected_fraction": float(fraction),
                "selected_count": int(selected_count),
                "captured_positive_count": captured_positive_count,
                "capture_rate": 0.0 if total_positives == 0 else float(captured_positive_count) / total_positives,
                "precision_at_k": observed_positive_rate,
                "lift_at_k": 0.0 if base_rate == 0 else observed_positive_rate / base_rate,
                "average_predicted_probability": float(selected["score"].mean()) if selected_count else 0.0,
                "observed_positive_rate": observed_positive_rate,
            }
        )
    return rows


def calibration_summary(y_true, y_prob) -> dict[str, float | int]:
    labels = np.asarray(y_true).astype(int)
    probabilities = np.asarray(y_prob).astype(float)
    mean_prediction = float(probabilities.mean()) if len(probabilities) else 0.0
    observed_rate = float(labels.mean()) if len(labels) else 0.0
    return {
        "row_count": int(len(labels)),
        "mean_prediction": mean_prediction,
        "observed_rate": observed_rate,
        "absolute_calibration_gap": abs(mean_prediction - observed_rate),
        "brier_score": float(brier_score_loss(labels, probabilities)) if len(labels) else math.nan,
    }


def calibration_by_decile(y_true, y_prob) -> list[dict[str, float | int]]:
    df = pd.DataFrame({"label": np.asarray(y_true).astype(int), "score": np.asarray(y_prob).astype(float)})
    if df.empty:
        return []
    df["score_decile"] = pd.qcut(df["score"].rank(method="first"), q=10, labels=False) + 1
    rows = []
    for decile, group in df.groupby("score_decile", sort=True):
        mean_prediction = float(group["score"].mean())
        observed_rate = float(group["label"].mean())
        rows.append(
            {
                "score_decile": int(decile),
                "row_count": int(len(group)),
                "mean_prediction": mean_prediction,
                "observed_rate": observed_rate,
                "calibration_error": abs(mean_prediction - observed_rate),
            }
        )
    return rows
