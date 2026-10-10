import json

from ai_platform.database.connection import get_connection


MODEL_COLUMNS = """
    model_id,
    name,
    model_type,
    version,
    provider,
    task_type,
    framework,
    artifact_uri,
    status,
    metrics_json,
    parameters_json,
    created_at
"""


class ModelRepository:
    """Provide CRUD operations and metadata management for AI models."""

    @staticmethod
    def _serialize_json(value: dict[str, object] | None) -> str:
        """Serialize a dictionary for storage in SQLite."""
        return json.dumps(
            value if value is not None else {},
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _to_model(row: tuple) -> dict[str, object]:
        """Convert a database row into a model registry record."""
        return {
            "model_id": row[0],
            "name": row[1],
            "model_type": row[2],
            "version": row[3],
            "provider": row[4],
            "task_type": row[5],
            "framework": row[6],
            "artifact_uri": row[7],
            "status": row[8],
            "metrics": json.loads(row[9]),
            "parameters": json.loads(row[10]),
            "created_at": row[11],
        }

    def create_model(
        self,
        name: str,
        model_type: str,
        version: str,
        *,
        provider: str = "local",
        task_type: str = "unknown",
        framework: str = "unknown",
        artifact_uri: str | None = None,
        status: str = "registered",
        metrics: dict[str, object] | None = None,
        parameters: dict[str, object] | None = None,
    ) -> int:
        """Register an AI model and return its ID."""
        connection = get_connection()

        try:
            cursor = connection.execute(
                """
                INSERT INTO models (
                    name, model_type, version, provider, task_type,
                    framework, artifact_uri, status, metrics_json,
                    parameters_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    model_type,
                    version,
                    provider,
                    task_type,
                    framework,
                    artifact_uri,
                    status,
                    self._serialize_json(metrics),
                    self._serialize_json(parameters),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)
        finally:
            connection.close()

    def get_model(self, model_id: int) -> dict[str, object] | None:
        """Return a registered model by ID."""
        connection = get_connection()

        try:
            row = connection.execute(
                f"SELECT {MODEL_COLUMNS} FROM models WHERE model_id = ?",
                (model_id,),
            ).fetchone()

            return None if row is None else self._to_model(row)
        finally:
            connection.close()

    def get_models(self) -> list[dict[str, object]]:
        """Return all registered models."""
        connection = get_connection()

        try:
            rows = connection.execute(
                f"SELECT {MODEL_COLUMNS} FROM models ORDER BY model_id"
            ).fetchall()

            return [self._to_model(row) for row in rows]
        finally:
            connection.close()

    def update_model(
        self,
        model_id: int,
        name: str,
        model_type: str,
        version: str,
        *,
        provider: str | None = None,
        task_type: str | None = None,
        framework: str | None = None,
        artifact_uri: str | None = None,
        status: str | None = None,
        metrics: dict[str, object] | None = None,
        parameters: dict[str, object] | None = None,
    ) -> None:
        """Update model information while preserving omitted metadata."""
        updates = [
            "name = ?",
            "model_type = ?",
            "version = ?",
        ]
        values: list[object] = [name, model_type, version]

        optional_fields = {
            "provider": provider,
            "task_type": task_type,
            "framework": framework,
            "artifact_uri": artifact_uri,
            "status": status,
        }

        for column, value in optional_fields.items():
            if value is not None:
                updates.append(f"{column} = ?")
                values.append(value)

        if metrics is not None:
            updates.append("metrics_json = ?")
            values.append(self._serialize_json(metrics))

        if parameters is not None:
            updates.append("parameters_json = ?")
            values.append(self._serialize_json(parameters))

        values.append(model_id)
        connection = get_connection()

        try:
            connection.execute(
                f"UPDATE models SET {', '.join(updates)} WHERE model_id = ?",
                values,
            )
            connection.commit()
        finally:
            connection.close()

    def delete_model(self, model_id: int) -> None:
        """Delete a registered model."""
        connection = get_connection()

        try:
            connection.execute(
                "DELETE FROM models WHERE model_id = ?",
                (model_id,),
            )
            connection.commit()
        finally:
            connection.close()
