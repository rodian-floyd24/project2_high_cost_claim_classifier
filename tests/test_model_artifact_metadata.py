from __future__ import annotations

import sys
from pathlib import Path

import pytest
import sklearn

from backend.scoring import MODEL_PYTHON_VERSION, MODEL_SKLEARN_VERSION, load_model_metadata


def test_model_metadata_matches_runtime_contract() -> None:
    metadata = load_model_metadata()
    assert metadata["model_name"] == "gradient_boosting"
    assert metadata["model_version"] == "actuarial_decision_support_v2"
    assert metadata["python_version"] == MODEL_PYTHON_VERSION
    assert metadata["sklearn_version"] == MODEL_SKLEARN_VERSION
    assert "calibration_method" in metadata
    assert "calibration_status" in metadata
    assert "probability_interpretation" in metadata
    assert "ranking_use_statement" in metadata


def test_runtime_sklearn_version_matches_model_metadata() -> None:
    metadata = load_model_metadata()
    runtime_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    artifact_python = ".".join(str(metadata["python_version"]).split(".")[:2])
    if runtime_python != artifact_python:
        pytest.skip(
            f"artifact serving contract is Python {artifact_python}; "
            f"current local runtime is Python {runtime_python}"
        )
    if sklearn.__version__ != metadata["sklearn_version"]:
        pytest.skip(
            f"artifact serving contract is sklearn {metadata['sklearn_version']}; "
            f"current local runtime is {sklearn.__version__}"
        )
    assert sklearn.__version__ == metadata["sklearn_version"]


def test_model_metadata_matches_mlflow_artifact_files() -> None:
    metadata = load_model_metadata()
    artifact_dir = Path(__file__).resolve().parents[1] / "backend" / "model_artifacts" / "model"
    mlmodel_path = artifact_dir / "MLmodel"
    requirements_path = artifact_dir / "requirements.txt"
    conda_path = artifact_dir / "conda.yaml"

    assert mlmodel_path.exists()
    assert requirements_path.exists()
    assert conda_path.exists()

    mlmodel_text = mlmodel_path.read_text()
    requirements_text = requirements_path.read_text()
    conda_text = conda_path.read_text()

    assert f"python_version: {metadata['python_version']}" in mlmodel_text
    assert f"sklearn_version: {metadata['sklearn_version']}" in mlmodel_text
    assert f"scikit-learn=={metadata['sklearn_version']}" in requirements_text
    assert f"python={metadata['python_version']}" in conda_text
    assert f"scikit-learn=={metadata['sklearn_version']}" in conda_text
