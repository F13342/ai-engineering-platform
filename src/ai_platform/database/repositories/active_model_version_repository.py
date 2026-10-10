"""Persist active model version selections."""

from ai_platform.database.connection import get_connection


class ActiveModelVersionRepository:
    """Store and retrieve the selected model ID for each model name."""

    @staticmethod
    def _validate_name(model_name: str) -> None:
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

    def set_active_version(
        self,
        model_name: str,
        model_id: int,
    ) -> dict[str, object]:
        """Select a registered version for a model name."""
        self._validate_name(model_name)
        self._validate_model_id(model_id)

        connection = get_connection()

        try:
            model = connection.execute(
                "SELECT name FROM models WHERE model_id = ?",
                (model_id,),
            ).fetchone()

            if model is None:
                raise LookupError(f"Model not found: {model_id}")

            if model[0] != model_name:
                raise ValueError(
                    "The selected model ID does not belong to this model name."
                )

            connection.execute(
                """
                INSERT INTO active_model_versions (model_name, model_id)
                VALUES (?, ?)
                ON CONFLICT(model_name) DO UPDATE SET
                    model_id = excluded.model_id,
                    activated_at = CURRENT_TIMESTAMP
                """,
                (model_name, model_id),
            )
            connection.commit()
        finally:
            connection.close()

        result = self.get_active_version(model_name)
        if result is None:
            raise RuntimeError(
                f"Active model version could not be retrieved: {model_name}"
            )

        return result

    def get_active_version(
        self,
        model_name: str,
    ) -> dict[str, object] | None:
        """Return the selected model ID for a name, if one is configured."""
        self._validate_name(model_name)
        connection = get_connection()

        try:
            row = connection.execute(
                """
                SELECT model_name, model_id, activated_at
                FROM active_model_versions
                WHERE model_name = ?
                """,
                (model_name,),
            ).fetchone()

            if row is None:
                return None

            return {
                "model_name": row[0],
                "model_id": row[1],
                "activated_at": row[2],
            }
        finally:
            connection.close()

    def get_active_versions(self) -> list[dict[str, object]]:
        """Return all active model version selections."""
        connection = get_connection()

        try:
            rows = connection.execute(
                """
                SELECT model_name, model_id, activated_at
                FROM active_model_versions
                ORDER BY model_name
                """
            ).fetchall()

            return [
                {
                    "model_name": row[0],
                    "model_id": row[1],
                    "activated_at": row[2],
                }
                for row in rows
            ]
        finally:
            connection.close()
