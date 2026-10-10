"""Coordinate model metadata and artifact storage."""

from ai_platform.database.repositories.model_repository import ModelRepository
from ai_platform.machine_learning.artifacts.store import ModelArtifactStore


class ModelRegistryService:
    """Register, retrieve, and load AI models."""

    def __init__(
        self,
        repository: ModelRepository | None = None,
        artifact_store: ModelArtifactStore | None = None,
    ) -> None:
        self.repository = repository or ModelRepository()
        self.artifact_store = artifact_store or ModelArtifactStore()

    def register_model(
        self,
        model: object,
        *,
        name: str,
        model_type: str,
        version: str,
        provider: str = "local",
        task_type: str = "unknown",
        framework: str = "unknown",
        metrics: dict[str, object] | None = None,
        parameters: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Register a model and persist its artifact."""
        model_id = self.repository.create_model(
            name=name,
            model_type=model_type,
            version=version,
            provider=provider,
            task_type=task_type,
            framework=framework,
            status="registered",
            metrics=metrics,
            parameters=parameters,
        )

        try:
            artifact_path = self.artifact_store.save(
                model,
                model_id=model_id,
                version=version,
            )

            self.repository.update_model(
                model_id=model_id,
                name=name,
                model_type=model_type,
                version=version,
                artifact_uri=artifact_path.relative_to(
                    self.artifact_store.root.resolve()
                ).as_posix(),
                status="ready",
            )
        except Exception:
            try:
                self.repository.update_model(
                    model_id=model_id,
                    name=name,
                    model_type=model_type,
                    version=version,
                    status="failed",
                )
            except Exception:
                pass
            raise

        record = self.repository.get_model(model_id)
        if record is None:
            raise RuntimeError(
                f"Registered model could not be retrieved: {model_id}"
            )

        return record

    def get_model(self, model_id: int) -> dict[str, object] | None:
        """Retrieve model metadata without loading the artifact."""
        return self.repository.get_model(model_id)

    def load_model(self, model_id: int) -> object:
        """Load a registered model only when its status is ready."""
        record = self.repository.get_model(model_id)

        if record is None:
            raise LookupError(f"Model not found: {model_id}")

        if record["status"] != "ready":
            raise ValueError(
                f"Model is not ready for loading: {model_id}"
            )

        artifact_uri = record["artifact_uri"]
        if not isinstance(artifact_uri, str) or not artifact_uri:
            raise ValueError(
                f"Model has no artifact URI: {model_id}"
            )

        return self.artifact_store.load(artifact_uri)
