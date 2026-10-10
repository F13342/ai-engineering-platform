from ai_platform.database.repositories.model_repository import ModelRepository
from ai_platform.database.schema import initialize_schema


def test_model_crud() -> None:
    """Test create, read, update, list, and delete for AI models."""
    initialize_schema()
    repository = ModelRepository()

    model_id = repository.create_model(
        name="AI Platform Base Model",
        model_type="language",
        version="0.1.0",
    )

    model = repository.get_model(model_id)

    assert model is not None
    assert model["name"] == "AI Platform Base Model"
    assert model["model_type"] == "language"
    assert model["version"] == "0.1.0"
    assert model["provider"] == "local"
    assert model["status"] == "registered"
    assert model["metrics"] == {}
    assert model["parameters"] == {}

    models = repository.get_models()
    assert any(item["model_id"] == model_id for item in models)

    repository.update_model(
        model_id=model_id,
        name="AI Platform Language Model",
        model_type="language",
        version="0.2.0",
    )

    updated_model = repository.get_model(model_id)
    assert updated_model is not None
    assert updated_model["name"] == "AI Platform Language Model"
    assert updated_model["version"] == "0.2.0"

    repository.delete_model(model_id)
    assert repository.get_model(model_id) is None


def test_model_registry_persists_metadata() -> None:
    """Test storing, reading, and updating model metadata."""
    initialize_schema()
    repository = ModelRepository()

    model_id = repository.create_model(
        name="Platform Text Model",
        model_type="language",
        version="1.0.0",
        provider="local",
        task_type="text-generation",
        framework="scikit-learn",
        artifact_uri="artifacts/text-model.joblib",
        status="ready",
        metrics={"accuracy": 0.98, "latency_ms": 12.4},
        parameters={"temperature": 0.2, "max_tokens": 512},
    )

    model = repository.get_model(model_id)
    assert model is not None
    assert model["provider"] == "local"
    assert model["task_type"] == "text-generation"
    assert model["framework"] == "scikit-learn"
    assert model["artifact_uri"] == "artifacts/text-model.joblib"
    assert model["status"] == "ready"
    assert model["metrics"] == {"accuracy": 0.98, "latency_ms": 12.4}
    assert model["parameters"] == {
        "temperature": 0.2,
        "max_tokens": 512,
    }

    repository.update_model(
        model_id=model_id,
        name="Platform Text Model",
        model_type="language",
        version="1.1.0",
        status="registered",
        metrics={"accuracy": 0.99},
    )

    updated = repository.get_model(model_id)
    assert updated is not None
    assert updated["version"] == "1.1.0"
    assert updated["status"] == "registered"
    assert updated["metrics"] == {"accuracy": 0.99}
    assert updated["parameters"] == {
        "temperature": 0.2,
        "max_tokens": 512,
    }

    repository.delete_model(model_id)
