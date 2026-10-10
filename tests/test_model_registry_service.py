from pathlib import Path

import pytest

from ai_platform.database.repositories.model_repository import ModelRepository
from ai_platform.database.schema import initialize_schema
from ai_platform.machine_learning.artifacts.store import ModelArtifactStore
from ai_platform.machine_learning.registry import ModelRegistryService


def test_registry_registers_and_loads_model(tmp_path) -> None:
    initialize_schema()
    repository = ModelRepository()
    service = ModelRegistryService(
        repository=repository,
        artifact_store=ModelArtifactStore(tmp_path / "models"),
    )

    original = {"weights": [0.2, 0.8], "bias": 0.1}
    model_id = None

    try:
        record = service.register_model(
            original,
            name="Platform Internal Model",
            model_type="general",
            version="0.1.0",
            task_type="inference",
            framework="joblib",
            metrics={"latency_ms": 12.4},
            parameters={"mode": "local"},
        )
        model_id = int(record["model_id"])

        assert record["status"] == "ready"
        assert record["artifact_uri"]
        assert not Path(str(record["artifact_uri"])).is_absolute()
        assert record["artifact_uri"] == f"{model_id}/0.1.0.joblib"
        assert record["task_type"] == "inference"
        assert record["framework"] == "joblib"
        assert record["metrics"] == {"latency_ms": 12.4}
        assert record["parameters"] == {"mode": "local"}
        assert service.load_model(model_id) == original

    finally:
        if model_id is not None:
            repository.delete_model(model_id)


def test_registry_rejects_loading_model_that_is_not_ready(
    tmp_path,
) -> None:
    initialize_schema()
    repository = ModelRepository()
    service = ModelRegistryService(
        repository=repository,
        artifact_store=ModelArtifactStore(tmp_path / "models"),
    )

    model_id = repository.create_model(
        name="Unready Model",
        model_type="general",
        version="0.1.0",
        status="registered",
    )

    try:
        with pytest.raises(ValueError, match="not ready"):
            service.load_model(model_id)
    finally:
        repository.delete_model(model_id)


def test_registry_reports_unknown_model(tmp_path) -> None:
    initialize_schema()
    service = ModelRegistryService(
        artifact_store=ModelArtifactStore(tmp_path / "models"),
    )

    with pytest.raises(LookupError, match="Model not found"):
        service.load_model(-99999)
