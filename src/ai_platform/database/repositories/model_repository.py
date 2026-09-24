from ai_platform.database.connection import get_connection


class ModelRepository:
    """Provide CRUD operations for AI models."""

    def create_model(
        self,
        name: str,
        model_type: str,
        version: str,
    ) -> int:
        """Create an AI model and return its ID."""
        connection = get_connection()

        try:
            cursor = connection.execute(
                """
                INSERT INTO models (name, model_type, version)
                VALUES (?, ?, ?)
                """,
                (name, model_type, version),
            )
            connection.commit()
            return int(cursor.lastrowid)
        finally:
            connection.close()

    def get_model(
        self,
        model_id: int,
    ) -> dict[str, object] | None:
        """Return an AI model by ID."""
        connection = get_connection()

        try:
            row = connection.execute(
                """
                SELECT
                    model_id,
                    name,
                    model_type,
                    version,
                    created_at
                FROM models
                WHERE model_id = ?
                """,
                (model_id,),
            ).fetchone()

            if row is None:
                return None

            return {
                "model_id": row[0],
                "name": row[1],
                "model_type": row[2],
                "version": row[3],
                "created_at": row[4],
            }
        finally:
            connection.close()

    def get_models(self) -> list[dict[str, object]]:
        """Return all AI models."""
        connection = get_connection()

        try:
            rows = connection.execute(
                """
                SELECT
                    model_id,
                    name,
                    model_type,
                    version,
                    created_at
                FROM models
                ORDER BY model_id
                """
            ).fetchall()

            return [
                {
                    "model_id": row[0],
                    "name": row[1],
                    "model_type": row[2],
                    "version": row[3],
                    "created_at": row[4],
                }
                for row in rows
            ]
        finally:
            connection.close()

    def update_model(
        self,
        model_id: int,
        name: str,
        model_type: str,
        version: str,
    ) -> None:
        """Update an AI model."""
        connection = get_connection()

        try:
            connection.execute(
                """
                UPDATE models
                SET name = ?, model_type = ?, version = ?
                WHERE model_id = ?
                """,
                (name, model_type, version, model_id),
            )
            connection.commit()
        finally:
            connection.close()

    def delete_model(self, model_id: int) -> None:
        """Delete an AI model."""
        connection = get_connection()

        try:
            connection.execute(
                """
                DELETE FROM models
                WHERE model_id = ?
                """,
                (model_id,),
            )
            connection.commit()
        finally:
            connection.close()
