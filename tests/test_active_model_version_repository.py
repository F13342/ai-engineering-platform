import pytest

from ai_platform.database.repositories.active_model_version_repository import (
    ActiveModelVersionRepository,
)
from ai_platform.database.repositories.model_repository import ModelRepository
from ai_platform.database.schema import initialize_schema


@pytest.fixture
def repositories():
    initialize_schema()
    return ActiveModelVersionRepository(), ModelRepository()


def create_model(repository, name, version):
    return repository.create_model(
        name=name,
        model_type="classifier",
        version=version,
        status="ready",
    )


def test_set_and_get_active_version(repositories):
    active_repository, model_repository = repositories
    name = "active-version-set-get-test"
    model_id = create_model(model_repository, name, "1.0.0")

    try:
        active = active_repository.set_active_version(name, model_id)
        retrieved = active_repository.get_active_version(name)

        assert active["model_name"] == name
        assert active["model_id"] == model_id
        assert active["activated_at"]
        assert retrieved == active
    finally:
        model_repository.delete_model(model_id)


def test_switch_active_version_for_same_name(repositories):
    active_repository, model_repository = repositories
    name = "active-version-switch-test"
    first_id = create_model(model_repository, name, "1.0.0")
    second_id = create_model(model_repository, name, "2.0.0")

    try:
        active_repository.set_active_version(name, first_id)
        updated = active_repository.set_active_version(name, second_id)

        assert updated["model_id"] == second_id
        assert active_repository.get_active_version(name)["model_id"] == second_id

        matching = [
            item
            for item in active_repository.get_active_versions()
            if item["model_name"] == name
        ]
        assert len(matching) == 1
        assert matching[0]["model_id"] == second_id
    finally:
        model_repository.delete_model(first_id)
        model_repository.delete_model(second_id)


def test_get_active_version_returns_none_when_not_configured(repositories):
    active_repository, _ = repositories

    assert active_repository.get_active_version(
        "no-active-version-configured"
    ) is None


def test_get_active_versions_returns_sorted_assignments(repositories):
    active_repository, model_repository = repositories
    first_name = "active-list-alpha"
    second_name = "active-list-beta"
    first_id = create_model(model_repository, first_name, "1.0.0")
    second_id = create_model(model_repository, second_name, "1.0.0")

    try:
        active_repository.set_active_version(second_name, second_id)
        active_repository.set_active_version(first_name, first_id)

        results = active_repository.get_active_versions()
        matching_names = [
            item["model_name"]
            for item in results
            if item["model_name"] in {first_name, second_name}
        ]

        assert matching_names == [first_name, second_name]
    finally:
        model_repository.delete_model(first_id)
        model_repository.delete_model(second_id)


def test_cannot_set_active_version_for_unknown_model(repositories):
    active_repository, _ = repositories

    with pytest.raises(LookupError, match="Model not found"):
        active_repository.set_active_version(
            "unknown-active-model",
            999999999,
        )


def test_model_name_must_match_registered_model(repositories):
    active_repository, model_repository = repositories
    model_id = create_model(
        model_repository,
        "registered-active-model",
        "1.0.0",
    )

    try:
        with pytest.raises(ValueError, match="does not belong"):
            active_repository.set_active_version(
                "different-model-name",
                model_id,
            )

        assert active_repository.get_active_version(
            "different-model-name"
        ) is None
    finally:
        model_repository.delete_model(model_id)


@pytest.mark.parametrize("name", ["", " ", None, 123])
def test_rejects_invalid_model_names(repositories, name):
    active_repository, _ = repositories

    with pytest.raises(ValueError, match="non-empty string"):
        active_repository.get_active_version(name)


@pytest.mark.parametrize("model_id", [0, -1, True, "1"])
def test_rejects_invalid_model_ids(repositories, model_id):
    active_repository, _ = repositories

    with pytest.raises(ValueError, match="positive integer"):
        active_repository.set_active_version(
            "active-version-invalid-id",
            model_id,
        )
