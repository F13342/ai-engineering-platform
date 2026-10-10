"""Manage active model version selections safely."""

from ai_platform.database.repositories.active_model_version_repository import (
    ActiveModelVersionRepository,
)
from ai_platform.database.repositories.model_repository import ModelRepository
from ai_platform.machine_learning.artifacts.store import ModelArtifactStore


class ActiveModelVersionService:
    """Select and retrieve the active ready version for a model name."""

    def __init__(
        self,
        active_repository: ActiveModelVersionRepository | None = None,
        model_repository: ModelRepository | None = None,
        artifact_store: ModelArtifactStore | None = None,
    ) -> None:
        self.active_repository = (
            active_repository
            if active_repository is not None
            else ActiveModelVersionRepository()
        )
        self.model_repository = (
            model_repository
            if model_repository is not None
            else ModelRepository()
        )
        self.artifact_store = (
            artifact_store
            if artifact_store is not None
            else ModelArtifactStore()
        )

    @staticmethod
    def _validate_model_name(model_name: str) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Model name must be a non-empty string.")

    @staticmethod
    def _validate_model_id(model_id: int) -> None:
        if (
            isinstance(model_id, bool)
            or not isinstance(model_id, int)
            or model_id <= 0
        ):
            raise ValueError("model_id must be a positive integer.")

    def activate_version(
        self,
        model_name: str,
        model_id: int,
    ) -> dict[str, object]:
        """Activate a model version only when it is ready and its artifact exists."""
        self._validate_model_name(model_name)
        self._validate_model_id(model_id)

        record = self.model_repository.get_model(model_id)

        if record is None:
            raise LookupError(f"Model not found: {model_id}")

        if record["name"] != model_name:
            raise ValueError(
                "The selected model ID does not belong to this model name."
            )

        if record["status"] != "ready":
            raise ValueError("Only ready models can be activated.")

        artifact_uri = record["artifact_uri"]
        if not isinstance(artifact_uri, str) or not artifact_uri:
            raise ValueError("Model has no artifact URI and cannot be activated.")

        if not self.artifact_store.exists(artifact_uri):
            raise ValueError(
                "Model artifact does not exist and cannot be activated."
            )

        active = self.active_repository.set_active_version(
            model_name,
            model_id,
        )

        return {
            "model_name": model_name,
            "model_id": model_id,
            "version": record["version"],
            "status": record["status"],
            "activated_at": active["activated_at"],
            "metrics": record["metrics"],
            "parameters": record["parameters"],
        }

    def get_active_version(
        self,
        model_name: str,
    ) -> dict[str, object] | None:
        """Retrieve the active version and its current model metadata."""
        self._validate_model_name(model_name)

        active = self.active_repository.get_active_version(model_name)
        if active is None:
            return None

        record = self.model_repository.get_model(active["model_id"])
        if record is None:
            raise RuntimeError(
                f"Active model record is missing: {active['model_id']}"
            )

        if record["name"] != model_name:
            raise RuntimeError(
                "Active model selection does not match the registered model name."
            )

        return {
            **active,
            "version": record["version"],
            "status": record["status"],
            "artifact_uri": record["artifact_uri"],
            "metrics": record["metrics"],
            "parameters": record["parameters"],
        }
