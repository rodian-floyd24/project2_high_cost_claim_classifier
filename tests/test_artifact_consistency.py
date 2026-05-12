from __future__ import annotations

import json
from pathlib import Path

import yaml

from shared.feature_contract import SERVED_MODEL_FEATURE_ORDER

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "backend" / "model_artifacts" / "model"
METADATA_PATH = Path(__file__).resolve().parents[1] / "backend" / "model_artifacts" / "model_metadata.json"
REQUIREMENTS_PATH = Path(__file__).resolve().parents[1] / "backend" / "requirements.txt"


def test_model_metadata_matches_mlmodel() -> None:
    metadata = json.loads(METADATA_PATH.read_text())
    mlmodel = yaml.safe_load((ARTIFACT_DIR / "MLmodel").read_text())

    assert metadata["sklearn_version"] == str(mlmodel["flavors"]["sklearn"]["sklearn_version"])
    assert metadata["python_version"] == str(mlmodel["flavors"]["python_function"]["python_version"])


def test_serving_requirements_match_model_metadata() -> None:
    metadata = json.loads(METADATA_PATH.read_text())
    backend_reqs = REQUIREMENTS_PATH.read_text()
    serving_reqs = (Path(__file__).resolve().parents[1] / "requirements-serving-py311.txt").read_text()

    assert f"scikit-learn=={metadata['sklearn_version']}" in backend_reqs
    assert f"scikit-learn=={metadata['sklearn_version']}" in serving_reqs


def test_model_signature_has_no_target_columns() -> None:
    from mlflow.models import Model
    model = Model.load(ARTIFACT_DIR)
    signature_features = [item.name for item in model.signature.inputs.inputs]

    forbidden_prefixes = ("target_", "next_year_")
    forbidden_columns = {"label", "target_annual_claim_cost"}

    for feature in signature_features:
        assert feature not in forbidden_columns
        assert not feature.startswith(forbidden_prefixes)


def test_artifact_signature_matches_served_feature_order() -> None:
    from mlflow.models import Model
    model = Model.load(ARTIFACT_DIR)
    signature_features = [item.name for item in model.signature.inputs.inputs]

    assert signature_features == SERVED_MODEL_FEATURE_ORDER
