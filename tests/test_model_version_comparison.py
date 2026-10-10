import pytest

from ai_platform.machine_learning.versions import ModelVersionComparisonService


class FakeModelRepository:
    def __init__(self, records=None):
        self.records = [] if records is None else records
        self.requested_name = None

    def get_models_by_name(self, name):
        self.requested_name = name
        return [
            record for record in self.records
            if record["name"] == name
        ]


def make_record(
    model_id,
    name,
    version,
    *,
    status="ready",
    metrics=None,
    parameters=None,
):
    return {
        "model_id": model_id,
        "name": name,
        "model_type": "classifier",
        "version": version,
        "provider": "local",
        "task_type": "classification",
        "framework": "scikit-learn",
        "artifact_uri": f"{model_id}/{version}.joblib",
        "status": status,
        "metrics": {} if metrics is None else metrics,
        "parameters": {} if parameters is None else parameters,
        "created_at": f"2026-10-10 10:0{model_id}:00",
    }


def test_compares_all_versions_and_preserves_metadata():
    name = "platform-model"
    repository = FakeModelRepository(
        [
            make_record(
                1,
                name,
                "1.0.0",
                metrics={"accuracy": 0.85, "latency_ms": 12.0},
                parameters={"depth": 3},
            ),
            make_record(
                2,
                name,
                "2.0.0",
                status="disabled",
                metrics={"accuracy": 0.92, "latency_ms": 20.0},
                parameters={"depth": 5},
            ),
            make_record(3, "unrelated-model", "1.0.0"),
        ]
    )
    service = ModelVersionComparisonService(repository=repository)

    result = service.compare_versions(name)

    assert repository.requested_name == name
    assert result["model_name"] == name
    assert result["version_count"] == 2

    versions = result["versions"]
    assert [item["model_id"] for item in versions] == [1, 2]
    assert [item["version"] for item in versions] == ["1.0.0", "2.0.0"]
    assert versions[0]["metrics"] == {
        "accuracy": 0.85,
        "latency_ms": 12.0,
    }
    assert versions[1]["metrics"] == {
        "accuracy": 0.92,
        "latency_ms": 20.0,
    }
    assert versions[0]["parameters"] == {"depth": 3}
    assert versions[1]["parameters"] == {"depth": 5}
    assert versions[1]["status"] == "disabled"


def test_returns_empty_comparison_when_no_versions_exist():
    service = ModelVersionComparisonService(
        repository=FakeModelRepository()
    )

    result = service.compare_versions("missing-model")

    assert result == {
        "model_name": "missing-model",
        "version_count": 0,
        "versions": [],
    }


def test_does_not_rank_or_reorder_versions():
    name = "unordered-model"
    repository = FakeModelRepository(
        [
            make_record(8, name, "3.0.0", metrics={"score": 0.1}),
            make_record(4, name, "1.0.0", metrics={"score": 0.9}),
        ]
    )
    service = ModelVersionComparisonService(repository=repository)

    result = service.compare_versions(name)

    assert [item["version"] for item in result["versions"]] == [
        "3.0.0",
        "1.0.0",
    ]
    assert all("rank" not in item for item in result["versions"])


def test_rejects_invalid_name_through_repository():
    class RejectingRepository:
        def get_models_by_name(self, name):
            raise ValueError("Model name must be a non-empty string.")

    service = ModelVersionComparisonService(
        repository=RejectingRepository()
    )

    with pytest.raises(ValueError, match="non-empty string"):
        service.compare_versions(" ")
