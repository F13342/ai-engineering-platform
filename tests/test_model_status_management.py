import pytest

from ai_platform.machine_learning.artifacts.store import ModelArtifactStore
from ai_platform.machine_learning.registry import ModelRegistryService


class FakeRepository:
    def __init__(self, status="ready", artifact_uri="1/1.0.0.joblib"):
        self.record = {
            "model_id": 1,
            "name": "platform-model",
            "model_type": "classifier",
            "version": "1.0.0",
            "artifact_uri": artifact_uri,
            "status": status,
        }

    def get_model(self, model_id):
        if model_id != self.record["model_id"]:
            return None
        return dict(self.record)

    def update_model(
        self, model_id, name, model_type, version, **kwargs
    ):
        if model_id != self.record["model_id"]:
            return

        self.record.update(
            name=name,
            model_type=model_type,
            version=version,
        )

        for field, value in kwargs.items():
            if value is not None:
                self.record[field] = value


class FakeArtifactStore:
    def __init__(self, exists=True):
        self.artifact_exists = exists
        self.checked_uri = None

    def exists(self, artifact_uri):
        self.checked_uri = artifact_uri
        return self.artifact_exists


def make_registry(status="ready", artifact_uri="1/1.0.0.joblib", exists=True):
    repository = FakeRepository(status=status, artifact_uri=artifact_uri)
    artifact_store = FakeArtifactStore(exists=exists)
    registry = ModelRegistryService(
        repository=repository,
        artifact_store=artifact_store,
    )
    return registry, repository, artifact_store


def test_can_disable_ready_model():
    registry, repository, _ = make_registry()

    result = registry.set_model_status(1, "disabled")

    assert result["status"] == "disabled"
    assert repository.record["status"] == "disabled"


def test_can_enable_disabled_model_when_artifact_exists():
    registry, repository, artifact_store = make_registry(
        status="disabled",
        exists=True,
    )

    result = registry.set_model_status(1, "ready")

    assert result["status"] == "ready"
    assert repository.record["status"] == "ready"
    assert artifact_store.checked_uri == "1/1.0.0.joblib"


def test_cannot_enable_model_when_artifact_is_missing():
    registry, repository, _ = make_registry(
        status="disabled",
        exists=False,
    )

    with pytest.raises(ValueError, match="artifact does not exist"):
        registry.set_model_status(1, "ready")

    assert repository.record["status"] == "disabled"


def test_cannot_enable_model_without_artifact_uri():
    registry, repository, _ = make_registry(
        status="disabled",
        artifact_uri=None,
    )

    with pytest.raises(ValueError, match="no artifact URI"):
        registry.set_model_status(1, "ready")

    assert repository.record["status"] == "disabled"


def test_cannot_change_registered_model_status():
    registry, repository, _ = make_registry(status="registered")

    with pytest.raises(ValueError, match="Cannot change a model"):
        registry.set_model_status(1, "disabled")

    assert repository.record["status"] == "registered"


@pytest.mark.parametrize(
    "status", ["registered", "failed", "running", "unknown"]
)
def test_rejects_unsupported_target_status(status):
    registry, repository, _ = make_registry()

    with pytest.raises(ValueError, match="Status must be either"):
        registry.set_model_status(1, status)

    assert repository.record["status"] == "ready"


@pytest.mark.parametrize("model_id", [0, -1, True, "1"])
def test_rejects_invalid_model_id(model_id):
    registry, _, _ = make_registry()

    with pytest.raises(ValueError, match="positive integer"):
        registry.set_model_status(model_id, "disabled")


def test_missing_model_raises_lookup_error():
    registry, _, _ = make_registry()

    with pytest.raises(LookupError, match="Model not found"):
        registry.set_model_status(2, "disabled")


def test_artifact_store_detects_existing_and_missing_files(tmp_path):
    store = ModelArtifactStore(root=tmp_path / "models")
    artifact = tmp_path / "models" / "1" / "1.0.0.joblib"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"test artifact")

    assert store.exists("1/1.0.0.joblib") is True
    assert store.exists("1/missing.joblib") is False


def test_artifact_store_rejects_path_outside_root(tmp_path):
    store = ModelArtifactStore(root=tmp_path / "models")

    with pytest.raises(ValueError, match="inside the model store"):
        store.exists("../outside.joblib")
