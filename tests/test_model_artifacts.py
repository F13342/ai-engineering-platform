import pytest

from ai_platform.machine_learning.artifacts.store import ModelArtifactStore


def test_model_artifact_round_trip(tmp_path) -> None:
    store = ModelArtifactStore(tmp_path / "models")
    original = {"weights": [0.2, 0.8], "bias": 0.1}

    artifact_path = store.save(original, model_id=1, version="0.1.0")

    assert artifact_path.is_file()
    assert artifact_path.name == "0.1.0.joblib"
    assert store.load(artifact_path) == original


def test_artifact_store_rejects_invalid_model_id(tmp_path) -> None:
    store = ModelArtifactStore(tmp_path / "models")

    with pytest.raises(ValueError, match="model_id"):
        store.save({}, model_id=0, version="1.0.0")


def test_artifact_store_rejects_path_injection(tmp_path) -> None:
    store = ModelArtifactStore(tmp_path / "models")

    with pytest.raises(ValueError, match="path separators"):
        store.save({}, model_id=1, version="../outside")


def test_artifact_store_rejects_paths_outside_store(tmp_path) -> None:
    store = ModelArtifactStore(tmp_path / "models")

    with pytest.raises(ValueError, match="inside the model store"):
        store.load(tmp_path / "outside.joblib")


def test_artifact_store_reports_missing_file(tmp_path) -> None:
    store = ModelArtifactStore(tmp_path / "models")
    missing_path = tmp_path / "models" / "1" / "1.0.0.joblib"

    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        store.load(missing_path)
