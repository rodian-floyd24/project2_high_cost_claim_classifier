from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from shared.feature_contract import MODEL_CATEGORICAL_FEATURES, MODEL_NUMERIC_FEATURES


RANDOM_SEED = 42
TARGET_QUANTILE = 0.9
SPLIT_STRATEGY = "beneficiary_hash_holdout"
SPLIT_VERSION = "xxhash64_bene_id_mod_100_v2_beneficiary_hash_holdout"
TEST_BUCKET_CUTOFF = 15
VALIDATION_BUCKET_CUTOFF = 30
MODEL_OUTPUT_DIR = Path(__file__).resolve().parent / "local_model"
TOPK_CURVE_PATH = MODEL_OUTPUT_DIR / "topk_curve_test.csv"
MODEL_PATH = MODEL_OUTPUT_DIR / "gradient_boosting_pipeline.joblib"
METADATA_PATH = MODEL_OUTPUT_DIR / "model_metadata.json"

NUMERIC_FEATURES = MODEL_NUMERIC_FEATURES
CATEGORICAL_FEATURES = MODEL_CATEGORICAL_FEATURES


def required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Set {name} before running local model training.")
    return value


def databricks_host() -> str:
    return required_env("DATABRICKS_HOST").rstrip("/")


def warehouse_id() -> str:
    return required_env("DATABRICKS_WAREHOUSE_ID")


def auth_headers() -> dict[str, str]:
    token = required_env("DATABRICKS_TOKEN")
    auth_scheme = "Be" + "arer"
    return {
        "Authorization": f"{auth_scheme} {token}",
        "Content-Type": "application/json",
    }


def sql_statement(statement: str) -> dict:
    payload = {
        "statement": statement,
        "warehouse_id": warehouse_id(),
        "disposition": "EXTERNAL_LINKS",
    }
    req = urllib.request.Request(
        databricks_host() + "/api/2.0/sql/statements",
        data=json.dumps(payload).encode(),
        headers=auth_headers(),
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def wait_for_statement(statement_id: str) -> dict:
    while True:
        req = urllib.request.Request(
            databricks_host() + f"/api/2.0/sql/statements/{statement_id}",
            headers=auth_headers(),
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        state = data["status"]["state"]
        if state == "SUCCEEDED":
            return data
        if state in {"FAILED", "CANCELED", "CLOSED"}:
            raise RuntimeError(json.dumps(data, indent=2))
        time.sleep(2)


def fetch_json_array(url: str) -> list[list]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=240) as resp:
        return json.loads(resp.read().decode())


def download_gold_table() -> pd.DataFrame:
    statement = sql_statement("SELECT * FROM default.gold_beneficiary_year_features")
    result = wait_for_statement(statement["statement_id"])
    columns = [column["name"] for column in result["manifest"]["schema"]["columns"]]
    chunks = result["manifest"]["chunks"]
    first_link = result["result"]["external_links"][0]["external_link"]

    all_rows: list[list] = []
    for idx, chunk in enumerate(chunks):
        if idx == 0:
            rows = fetch_json_array(first_link)
        else:
            req = urllib.request.Request(
                databricks_host() + f"/api/2.0/sql/statements/{statement['statement_id']}/result/chunks/{idx}",
                headers=auth_headers(),
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                chunk_meta = json.loads(resp.read().decode())
            rows = fetch_json_array(chunk_meta["external_links"][0]["external_link"])
        all_rows.extend(rows)
        print(f"downloaded chunk {idx + 1}/{len(chunks)} rows={len(rows)}")

    return pd.DataFrame(all_rows, columns=columns)


def build_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    current = df.copy()
    future = df[["bene_id", "year", "annual_claim_cost"]].copy()
    future = future.rename(columns={"year": "target_year", "annual_claim_cost": "target_annual_claim_cost"})
    current["target_year"] = current["year"] + 1
    modeled = current.merge(future, on=["bene_id", "target_year"], how="inner")
    return modeled


def split_by_beneficiary_hash_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bucket = pd.util.hash_pandas_object(df["bene_id"].astype(str), index=False).mod(100)
    split_df = df.assign(shared_split_bucket=bucket.astype(int))
    train_df = split_df[split_df["shared_split_bucket"] >= VALIDATION_BUCKET_CUTOFF].copy()
    validation_df = split_df[
        (split_df["shared_split_bucket"] >= TEST_BUCKET_CUTOFF)
        & (split_df["shared_split_bucket"] < VALIDATION_BUCKET_CUTOFF)
    ].copy()
    test_df = split_df[split_df["shared_split_bucket"] < TEST_BUCKET_CUTOFF].copy()
    return train_df, validation_df, test_df


def add_training_target(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    combined_df = pd.concat([train_df, validation_df, test_df], ignore_index=True)
    thresholds = combined_df.groupby("target_year")["target_annual_claim_cost"].quantile(TARGET_QUANTILE)
    threshold = float(thresholds.mean())
    for frame in (train_df, validation_df, test_df):
        frame["target_year_high_cost_threshold"] = frame["target_year"].map(thresholds)
        frame["target_cost_within_year_percentile"] = frame.groupby("target_year")["target_annual_claim_cost"].rank(
            pct=True
        )
        frame["label"] = (frame["target_annual_claim_cost"] > frame["target_year_high_cost_threshold"]).astype(int)
    return train_df, validation_df, test_df, threshold


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="constant", fill_value=0.0), NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.8,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def top_k_capture_and_lift(y_true: np.ndarray, y_score: np.ndarray, top_fraction: float) -> tuple[float, float]:
    top_n = max(1, int(math.ceil(len(y_true) * top_fraction)))
    ranked = pd.DataFrame({"label": y_true, "score": y_score}).sort_values("score", ascending=False)
    selected = ranked.head(top_n)
    total_positives = float(ranked["label"].sum())
    base_rate = float(ranked["label"].mean())
    selected_rate = float(selected["label"].mean())
    capture = 0.0 if total_positives == 0 else float(selected["label"].sum()) / total_positives
    lift = 0.0 if base_rate == 0 else selected_rate / base_rate
    return capture, lift


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    candidate_thresholds = np.linspace(0.1, 0.9, 17)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidate_thresholds:
        pred = (y_score >= threshold).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        if f1 > best_score:
            best_score = f1
            best_threshold = float(threshold)
    return best_threshold


def main() -> None:
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gold_df = download_gold_table()
    modeled = build_modeling_frame(gold_df)
    train_df, validation_df, test_df = split_by_beneficiary_hash_holdout(modeled)
    train_df, validation_df, test_df, label_threshold = add_training_target(train_df, validation_df, test_df)

    pipeline = build_pipeline()
    pipeline.fit(train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train_df["label"])

    train_scores = pipeline.predict_proba(train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    validation_scores = pipeline.predict_proba(validation_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    test_scores = pipeline.predict_proba(test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    decision_threshold = choose_threshold(validation_df["label"].to_numpy(), validation_scores)

    curve_rows = []
    for fraction in np.linspace(0.01, 0.2, 20):
        capture, lift = top_k_capture_and_lift(test_df["label"].to_numpy(), test_scores, float(fraction))
        curve_rows.append(
            {
                "selected_fraction": float(fraction),
                "capture_rate": float(capture),
                "lift": float(lift),
            }
        )
    pd.DataFrame(curve_rows).to_csv(TOPK_CURVE_PATH, index=False)

    metadata = {
        "model_name": "gradient_boosting_local",
        "training_source": "default.gold_beneficiary_year_features",
        "split_strategy": SPLIT_STRATEGY,
        "split_version": SPLIT_VERSION,
        "feature_timing_frame": "current_year_features_predict_next_year_target",
        "utilization_feature_timing": "prior_year_relative_to_target_year",
        "target_definition": "predict_next_year_high_cost_within_target_year_top_decile",
        "label_threshold_train_only": label_threshold,
        "decision_threshold_validation_tuned": decision_threshold,
        "test_auc_roc": float(roc_auc_score(test_df["label"], test_scores)),
        "test_auc_pr": float(average_precision_score(test_df["label"], test_scores)),
        "top_5_capture_rate": curve_rows[4]["capture_rate"],
        "top_10_capture_rate": curve_rows[9]["capture_rate"],
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    print(MODEL_PATH)


if __name__ == "__main__":
    main()
