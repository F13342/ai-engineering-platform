"""Compare registered model versions without assuming metric direction."""

from ai_platform.database.repositories.model_repository import ModelRepository


class ModelVersionComparisonService:
    """Provide comparable metadata for all versions of a named model."""

    def __init__(
        self,
        repository: ModelRepository | None = None,
    ) -> None:
        self.repository = (
            repository if repository is not None else ModelRepository()
        )

    def compare_versions(self, name: str) -> dict[str, object]:
        """Return all exact-name versions and their metrics.

        Versions are returned in repository order. This service deliberately
        does not rank versions or decide which metric values are preferable.
        """
        records = self.repository.get_models_by_name(name)

        versions = [
            {
                "model_id": record["model_id"],
                "version": record["version"],
                "status": record["status"],
                "model_type": record["model_type"],
                "provider": record["provider"],
                "task_type": record["task_type"],
                "framework": record["framework"],
                "metrics": record["metrics"],
                "parameters": record["parameters"],
                "created_at": record["created_at"],
            }
            for record in records
        ]

        return {
            "model_name": name,
            "version_count": len(versions),
            "versions": versions,
        }
