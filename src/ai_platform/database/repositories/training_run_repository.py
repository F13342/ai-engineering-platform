"""Persist the lifecycle of model training attempts."""

import json

from ai_platform.database.connection import get_connection


TRAINING_RUN_COLUMNS = """
    run_id,
    model_id,
    model_name,
    model_type,
    model_version,
    status,
    metrics_json,
    parameters_json,
    error_message,
    started_at,
    finished_at
"""


class TrainingRunRepository:
    """Create, retrieve, complete, and fail training runs."""

    STATUSES = {"running", "completed", "failed"}

    @staticmethod
    def _serialize_json(value: dict[str, object] | None) -> str:
        """Serialize metadata as strict JSON."""
        if value is None:
            value = {}

        if not isinstance(value, dict):
            raise TypeError("Training metadata must be a dictionary.")

        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )

    @staticmethod
    def _to_training_run(row: tuple) -> dict[str, object]:
        """Convert a database row into a training run record."""
        return {
            "run_id": row[0],
            "model_id": row[1],
            "model_name": row[2],
            "model_type": row[3],
            "model_version": row[4],
            "status": row[5],
            "metrics": json.loads(row[6]),
            "parameters": json.loads(row[7]),
            "error_message": row[8],
            "started_at": row[9],
            "finished_at": row[10],
        }

    @staticmethod
    def _validate_run_id(run_id: int) -> None:
        if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id <= 0:
            raise ValueError("run_id must be a positive integer.")

    @classmethod
    def _validate_status(cls, status: str) -> None:
        if status not in cls.STATUSES:
            raise ValueError(
                f"Invalid training status: {status}. "
                f"Allowed statuses: {sorted(cls.STATUSES)}"
            )

    def create_training_run(
        self,
        *,
        model_name: str,
        model_type: str,
        model_version: str,
        parameters: dict[str, object] | None = None,
    ) -> int:
        """Create a training run in the running state."""
        for field_name, value in (
            ("model_name", model_name),
            ("model_type", model_type),
            ("model_version", model_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")

        parameters_json = self._serialize_json(parameters)
        connection = get_connection()

        try:
            cursor = connection.execute(
                """
                INSERT INTO training_runs (
                    model_name,
                    model_type,
                    model_version,
                    status,
                    parameters_json
                )
                VALUES (?, ?, ?, 'running', ?)
                """,
                (model_name, model_type, model_version, parameters_json),
            )
            connection.commit()
            return int(cursor.lastrowid)
        finally:
            connection.close()

    def get_training_run(self, run_id: int) -> dict[str, object] | None:
        """Retrieve a training run by ID."""
        self._validate_run_id(run_id)
        connection = get_connection()

        try:
            row = connection.execute(
                f"SELECT {TRAINING_RUN_COLUMNS} "
                "FROM training_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()

            return None if row is None else self._to_training_run(row)
        finally:
            connection.close()

    def get_training_runs(
        self,
        *,
        status: str | None = None,
        model_id: int | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        """List recent training runs, optionally filtering by status or model."""
        if status is not None:
            self._validate_status(status)

        if model_id is not None:
            if (
                isinstance(model_id, bool)
                or not isinstance(model_id, int)
                or model_id <= 0
            ):
                raise ValueError("model_id must be a positive integer.")

        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer.")

        conditions: list[str] = []
        values: list[object] = []

        if status is not None:
            conditions.append("status = ?")
            values.append(status)

        if model_id is not None:
            conditions.append("model_id = ?")
            values.append(model_id)

        where_clause = (
            f"WHERE {' AND '.join(conditions)}" if conditions else ""
        )

        values.append(limit)
        connection = get_connection()

        try:
            rows = connection.execute(
                f"""
                SELECT {TRAINING_RUN_COLUMNS}
                FROM training_runs
                {where_clause}
                ORDER BY run_id DESC
                LIMIT ?
                """,
                values,
            ).fetchall()

            return [self._to_training_run(row) for row in rows]
        finally:
            connection.close()

    def complete_training_run(
        self,
        run_id: int,
        *,
        model_id: int,
        metrics: dict[str, object] | None = None,
        parameters: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Mark a running attempt completed and link its registered model."""
        self._validate_run_id(run_id)

        if (
            isinstance(model_id, bool)
            or not isinstance(model_id, int)
            or model_id <= 0
        ):
            raise ValueError("model_id must be a positive integer.")

        metrics_json = self._serialize_json(metrics)
        parameters_json = (
            None if parameters is None else self._serialize_json(parameters)
        )
        connection = get_connection()

        try:
            cursor = connection.execute(
                """
                UPDATE training_runs
                SET status = 'completed',
                    model_id = ?,
                    metrics_json = ?,
                    parameters_json = COALESCE(?, parameters_json),
                    error_message = NULL,
                    finished_at = CURRENT_TIMESTAMP
                WHERE run_id = ? AND status = 'running'
                """,
                (model_id, metrics_json, parameters_json, run_id),
            )

            if cursor.rowcount == 0:
                connection.rollback()
            else:
                connection.commit()
        finally:
            connection.close()

        if cursor.rowcount == 0:
            existing = self.get_training_run(run_id)
            if existing is None:
                raise LookupError(f"Training run not found: {run_id}")
            raise ValueError("Only running training runs can be completed.")

        result = self.get_training_run(run_id)
        if result is None:
            raise RuntimeError(f"Completed training run could not be retrieved: {run_id}")

        return result

    def fail_training_run(
        self,
        run_id: int,
        *,
        error_message: str,
    ) -> dict[str, object]:
        """Mark a running attempt as failed and preserve its error details."""
        self._validate_run_id(run_id)

        if not isinstance(error_message, str) or not error_message.strip():
            raise ValueError("error_message must be a non-empty string.")

        connection = get_connection()

        try:
            cursor = connection.execute(
                """
                UPDATE training_runs
                SET status = 'failed',
                    error_message = ?,
                    finished_at = CURRENT_TIMESTAMP
                WHERE run_id = ? AND status = 'running'
                """,
                (error_message, run_id),
            )

            if cursor.rowcount == 0:
                connection.rollback()
            else:
                connection.commit()
        finally:
            connection.close()

        if cursor.rowcount == 0:
            existing = self.get_training_run(run_id)
            if existing is None:
                raise LookupError(f"Training run not found: {run_id}")
            raise ValueError("Only running training runs can be failed.")

        result = self.get_training_run(run_id)
        if result is None:
            raise RuntimeError(f"Failed training run could not be retrieved: {run_id}")

        return result
