import pytest

from ai_platform.database.repositories.model_repository import ModelRepository
from ai_platform.database.schema import initialize_schema


def test_get_models_by_name_returns_all_exact_matching_versions():
    initialize_schema()
    repository = ModelRepository()
    name = "version-comparison-test-model"

    first_id = repository.create_model(
        name=name,
        model_type="classifier",
        version="1.0.0",
        status="ready",
        metrics={"accuracy": 0.85},
        parameters={"depth": 3},
    )
    second_id = repository.create_model(
        name=name,
        model_type="classifier",
        version="2.0.0",
        status="disabled",
        metrics={"accuracy": 0.92},
        parameters={"depth": 5},
    )
    unrelated_id = repository.create_model(
        name="version-comparison-test-model-other",
        model_type="classifier",
        version="1.0.0",
    )

    try:
        versions = repository.get_models_by_name(name)

        assert [item["model_id"] for item in versions] == [
            first_id,
            second_id,
        ]
        assert [item["version"] for item in versions] == [
            "1.0.0",
            "2.0.0",
        ]
        assert versions[0]["metrics"] == {"accuracy": 0.85}
        assert versions[1]["metrics"] == {"accuracy": 0.92}
        assert versions[0]["parameters"] == {"depth": 3}
        assert versions[1]["parameters"] == {"depth": 5}
        assert all(item["model_id"] != unrelated_id for item in versions)
    finally:
        repository.delete_model(first_id)
        repository.delete_model(second_id)
        repository.delete_model(unrelated_id)


def test_get_models_by_name_returns_empty_list_when_not_found():
    initialize_schema()
    repository = ModelRepository()

    assert repository.get_models_by_name(
        "nonexistent-version-comparison-model"
    ) == []


@pytest.mark.parametrize("name", ["", " ", None, 123])
def test_get_models_by_name_rejects_invalid_name(name):
    initialize_schema()
    repository = ModelRepository()

    with pytest.raises(ValueError, match="non-empty string"):
        repository.get_models_by_name(name)
