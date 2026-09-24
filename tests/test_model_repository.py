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
