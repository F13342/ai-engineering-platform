import math

import pytest

from ai_platform.database.repositories.model_repository import ModelRepository
from ai_platform.database.repositories.training_run_repository import (
    TrainingRunRepository,
)
from ai_platform.database.schema import initialize_schema


@pytest.fixture
def repository():
    initialize_schema()
    return TrainingRunRepository()


def create_run(repository, *, name="platform-lifecycle-model", parameters=None):
    return repository.create_training_run(
        model_name=name,
        model_type="classifier",
        model_version="1.0.0",
        parameters=parameters,
    )


def create_registered_model():
    initialize_schema()
    return ModelRepository().create_model(
        name="platform-lifecycle-model",
        model_type="classifier",
        version="1.0.0",
        status="ready",
        metrics={"accuracy": 0.95},
    )


def test_create_training_run_starts_running(repository):
    run_id = create_run(
        repository,
        parameters={"learning_rate": 0.01, "epochs": 10},
    )

    run = repository.get_training_run(run_id)

    assert run is not None
    assert run["run_id"] == run_id
    assert run["model_id"] is None
    assert run["model_name"] == "platform-lifecycle-model"
    assert run["model_type"] == "classifier"
    assert run["model_version"] == "1.0.0"
    assert run["status"] == "running"
    assert run["metrics"] == {}
    assert run["parameters"] == {"learning_rate": 0.01, "epochs": 10}
    assert run["error_message"] is None
    assert run["started_at"]
    assert run["finished_at"] is None


def test_create_training_run_defaults_parameters_to_empty_dictionary(repository):
    run_id = create_run(repository)

    run = repository.get_training_run(run_id)

    assert run is not None
    assert run["parameters"] == {}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", ""),
        ("model_type", " "),
        ("model_version", ""),
    ],
)
def test_create_training_run_rejects_empty_required_fields(
    repository,
    field,
    value,
):
    values = {
        "model_name": "platform-model",
        "model_type": "classifier",
        "model_version": "1.0.0",
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        repository.create_training_run(**values)


def test_create_training_run_rejects_non_dictionary_parameters(repository):
    with pytest.raises(TypeError, match="dictionary"):
        repository.create_training_run(
            model_name="platform-model",
            model_type="classifier",
            model_version="1.0.0",
            parameters=["not", "a", "dictionary"],
        )


def test_create_training_run_rejects_non_finite_json_values(repository):
    with pytest.raises(ValueError):
        create_run(repository, parameters={"invalid": math.nan})


def test_get_training_run_returns_none_for_unknown_id(repository):
    assert repository.get_training_run(999999999) is None


@pytest.mark.parametrize("run_id", [0, -1, True, "1"])
def test_get_training_run_rejects_invalid_ids(repository, run_id):
    with pytest.raises(ValueError, match="positive integer"):
        repository.get_training_run(run_id)


def test_complete_training_run_links_model_and_saves_metrics(repository):
    run_id = create_run(repository)
    model_id = create_registered_model()

    run = repository.complete_training_run(
        run_id,
        model_id=model_id,
        metrics={"accuracy": 0.95, "f1": 0.92},
    )

    assert run["status"] == "completed"
    assert run["model_id"] == model_id
    assert run["metrics"] == {"accuracy": 0.95, "f1": 0.92}
    assert run["error_message"] is None
    assert run["finished_at"]


def test_complete_training_run_rejects_invalid_model_id(repository):
    run_id = create_run(repository)

    with pytest.raises(ValueError, match="model_id"):
        repository.complete_training_run(run_id, model_id=0)

    assert repository.get_training_run(run_id)["status"] == "running"


def test_complete_training_run_rejects_non_finite_metrics(repository):
    run_id = create_run(repository)
    model_id = create_registered_model()

    with pytest.raises(ValueError):
        repository.complete_training_run(
            run_id,
            model_id=model_id,
            metrics={"invalid": math.inf},
        )

    assert repository.get_training_run(run_id)["status"] == "running"


def test_fail_training_run_preserves_error_details(repository):
    run_id = create_run(repository)

    run = repository.fail_training_run(
        run_id,
        error_message="Training process failed.",
    )

    assert run["status"] == "failed"
    assert run["error_message"] == "Training process failed."
    assert run["finished_at"]
    assert run["model_id"] is None


def test_fail_training_run_rejects_empty_error_message(repository):
    run_id = create_run(repository)

    with pytest.raises(ValueError, match="non-empty string"):
        repository.fail_training_run(run_id, error_message=" ")

    assert repository.get_training_run(run_id)["status"] == "running"


def test_cannot_complete_a_run_twice(repository):
    run_id = create_run(repository)
    model_id = create_registered_model()

    repository.complete_training_run(run_id, model_id=model_id)

    with pytest.raises(ValueError, match="Only running"):
        repository.complete_training_run(run_id, model_id=model_id)


def test_cannot_fail_a_completed_run(repository):
    run_id = create_run(repository)
    model_id = create_registered_model()

    repository.complete_training_run(run_id, model_id=model_id)

    with pytest.raises(ValueError, match="Only running"):
        repository.fail_training_run(run_id, error_message="Late failure")


def test_cannot_complete_or_fail_unknown_run(repository):
    with pytest.raises(LookupError, match="not found"):
        repository.complete_training_run(999999999, model_id=1)

    with pytest.raises(LookupError, match="not found"):
        repository.fail_training_run(
            999999999,
            error_message="Unknown run",
        )


def test_list_training_runs_filters_by_status(repository):
    completed_id = create_run(repository, name="completed-lifecycle-model")
    failed_id = create_run(repository, name="failed-lifecycle-model")
    model_id = create_registered_model()

    repository.complete_training_run(completed_id, model_id=model_id)
    repository.fail_training_run(failed_id, error_message="Expected failure")

    completed_runs = repository.get_training_runs(status="completed")
    failed_runs = repository.get_training_runs(status="failed")

    completed_by_id = {run["run_id"]: run for run in completed_runs}
    failed_by_id = {run["run_id"]: run for run in failed_runs}

    assert completed_by_id[completed_id]["status"] == "completed"
    assert failed_by_id[failed_id]["status"] == "failed"
    assert all(run["status"] == "completed" for run in completed_runs)
    assert all(run["status"] == "failed" for run in failed_runs)


def test_list_training_runs_filters_by_model_id(repository):
    run_id = create_run(repository)
    model_id = create_registered_model()

    repository.complete_training_run(run_id, model_id=model_id)

    runs = repository.get_training_runs(model_id=model_id)

    assert runs
    assert all(run["model_id"] == model_id for run in runs)
    assert any(run["run_id"] == run_id for run in runs)


def test_list_training_runs_respects_limit(repository):
    for index in range(3):
        create_run(repository, name=f"limited-run-model-{index}")

    runs = repository.get_training_runs(limit=2)

    assert len(runs) == 2
    assert runs[0]["run_id"] > runs[1]["run_id"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"status": "unknown"}, "Invalid training status"),
        ({"limit": 0}, "limit must be a positive integer"),
        ({"limit": True}, "limit must be a positive integer"),
        ({"model_id": 0}, "model_id must be a positive integer"),
    ],
)
def test_list_training_runs_validates_filters(repository, kwargs, message):
    with pytest.raises(ValueError, match=message):
        repository.get_training_runs(**kwargs)
