import pytest

from ai_platform.machine_learning.versions import ActiveModelVersionService


class FakeActiveRepository:
    def __init__(self):
        self.active = {}

    def set_active_version(self, model_name, model_id):
        record = {
            "model_name": model_name,
            "model_id": model_id,
            "activated_at": "2026-10-10 12:00:00",
        }
        self.active[model_name] = record
        return record

    def get_active_version(self, model_name):
        return self.active.get(model_name)


class FakeModelRepository:
    def __init__(self, records=None):
        self.records = {} if records is None else records

    def get_model(self, model_id):
        record = self.records.get(model_id)
        return None if record is None else dict(record)


class FakeArtifactStore:
    def __init__(self, exists=True):
        self.artifact_exists = exists
        self.checked_uri = None

    def exists(self, artifact_uri):
        self.checked_uri = artifact_uri
        return self.artifact_exists


def make_record(
    model_id=1,
    name="platform-model",
    version="1.0.0",
    status="ready",
    artifact_uri="1/1.0.0.joblib",
):
    return {
        "model_id": model_id,
        "name": name,
        "model_type": "classifier",
        "version": version,
        "provider": "local",
        "task_type": "classification",
        "framework": "scikit-learn",
        "artifact_uri": artifact_uri,
        "status": status,
        "metrics": {"accuracy": 0.95},
        "parameters": {"depth": 3},
        "created_at": "2026-10-10 10:00:00",
    }


def make_service(records=None, artifact_exists=True):
    active_repository = FakeActiveRepository()
    model_repository = FakeModelRepository(records or {1: make_record()})
    artifact_store = FakeArtifactStore(exists=artifact_exists)

    service = ActiveModelVersionService(
        active_repository=active_repository,
        model_repository=model_repository,
        artifact_store=artifact_store,
    )

    return service, active_repository, model_repository, artifact_store


def test_activate_ready_version_with_existing_artifact():
    service, active_repository, _, artifact_store = make_service()

    result = service.activate_version("platform-model", 1)

    assert result["model_name"] == "platform-model"
    assert result["model_id"] == 1
    assert result["version"] == "1.0.0"
    assert result["status"] == "ready"
    assert result["metrics"] == {"accuracy": 0.95}
    assert result["activated_at"]
    assert artifact_store.checked_uri == "1/1.0.0.joblib"
    assert active_repository.active["platform-model"]["model_id"] == 1


def test_cannot_activate_unknown_model():
    service, active_repository, _, _ = make_service(records={})

    with pytest.raises(LookupError, match="Model not found"):
        service.activate_version("platform-model", 999)

    assert active_repository.active == {}


def test_cannot_activate_model_with_different_name():
    service, active_repository, _, _ = make_service()

    with pytest.raises(ValueError, match="does not belong"):
        service.activate_version("another-model", 1)

    assert active_repository.active == {}


def test_cannot_activate_model_that_is_not_ready():
    record = make_record(status="disabled")
    service, active_repository, _, artifact_store = make_service(
        records={1: record},
    )

    with pytest.raises(ValueError, match="Only ready models"):
        service.activate_version("platform-model", 1)

    assert active_repository.active == {}
    assert artifact_store.checked_uri is None


def test_cannot_activate_model_without_artifact_uri():
    record = make_record(artifact_uri=None)
    service, active_repository, _, artifact_store = make_service(
        records={1: record},
    )

    with pytest.raises(ValueError, match="no artifact URI"):
        service.activate_version("platform-model", 1)

    assert active_repository.active == {}
    assert artifact_store.checked_uri is None


def test_cannot_activate_model_when_artifact_is_missing():
    service, active_repository, _, _ = make_service(
        artifact_exists=False,
    )

    with pytest.raises(ValueError, match="artifact does not exist"):
        service.activate_version("platform-model", 1)

    assert active_repository.active == {}


@pytest.mark.parametrize("model_id", [0, -1, True, "1"])
def test_rejects_invalid_model_id(model_id):
    service, active_repository, _, _ = make_service()

    with pytest.raises(ValueError, match="positive integer"):
        service.activate_version("platform-model", model_id)

    assert active_repository.active == {}


@pytest.mark.parametrize("name", ["", " ", None, 123])
def test_rejects_invalid_model_name(name):
    service, active_repository, _, _ = make_service()

    with pytest.raises(ValueError, match="non-empty string"):
        service.activate_version(name, 1)

    assert active_repository.active == {}


def test_get_active_version_returns_none_when_not_configured():
    service, _, _, _ = make_service()

    assert service.get_active_version("platform-model") is None


def test_get_active_version_returns_current_model_metadata():
    service, active_repository, _, _ = make_service()
    active_repository.set_active_version("platform-model", 1)

    result = service.get_active_version("platform-model")

    assert result is not None
    assert result["model_name"] == "platform-model"
    assert result["model_id"] == 1
    assert result["version"] == "1.0.0"
    assert result["artifact_uri"] == "1/1.0.0.joblib"
    assert result["metrics"] == {"accuracy": 0.95}
    assert result["parameters"] == {"depth": 3}


def test_get_active_version_fails_if_registered_model_is_missing():
    service, active_repository, model_repository, _ = make_service()
    active_repository.set_active_version("platform-model", 1)
    model_repository.records.clear()

    with pytest.raises(RuntimeError, match="Active model record is missing"):
        service.get_active_version("platform-model")


def test_get_active_version_rejects_mismatched_model_name():
    service, active_repository, model_repository, _ = make_service()
    active_repository.set_active_version("platform-model", 1)
    model_repository.records[1]["name"] = "different-model"

    with pytest.raises(RuntimeError, match="does not match"):
        service.get_active_version("platform-model")
