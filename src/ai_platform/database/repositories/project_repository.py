from ai_platform.database.connection import get_connection


class ProjectRepository:
    """Provide CRUD operations for projects."""

    def create_project(
        self,
        user_id: int,
        name: str,
        description: str,
    ) -> int:
        """Create a project and return its ID."""
        connection = get_connection()

        try:
            cursor = connection.execute(
                """
                INSERT INTO projects (user_id, name, description)
                VALUES (?, ?, ?)
                """,
                (user_id, name, description),
            )
            connection.commit()
            return int(cursor.lastrowid)
        finally:
            connection.close()

    def get_project(self, project_id: int) -> dict[str, object] | None:
        """Return a project by ID."""
        connection = get_connection()

        try:
            row = connection.execute(
                """
                SELECT project_id, user_id, name, description, created_at
                FROM projects
                WHERE project_id = ?
                """,
                (project_id,),
            ).fetchone()

            if row is None:
                return None

            return {
                "project_id": row[0],
                "user_id": row[1],
                "name": row[2],
                "description": row[3],
                "created_at": row[4],
            }
        finally:
            connection.close()

    def get_projects_by_user(
        self,
        user_id: int,
    ) -> list[dict[str, object]]:
        """Return all projects belonging to a user."""
        connection = get_connection()

        try:
            rows = connection.execute(
                """
                SELECT project_id, user_id, name, description, created_at
                FROM projects
                WHERE user_id = ?
                ORDER BY project_id
                """,
                (user_id,),
            ).fetchall()

            return [
                {
                    "project_id": row[0],
                    "user_id": row[1],
                    "name": row[2],
                    "description": row[3],
                    "created_at": row[4],
                }
                for row in rows
            ]
        finally:
            connection.close()

    def update_project(
        self,
        project_id: int,
        name: str,
        description: str,
    ) -> None:
        """Update an existing project."""
        connection = get_connection()

        try:
            connection.execute(
                """
                UPDATE projects
                SET name = ?, description = ?
                WHERE project_id = ?
                """,
                (name, description, project_id),
            )
            connection.commit()
        finally:
            connection.close()

    def delete_project(self, project_id: int) -> None:
        """Delete a project by ID."""
        connection = get_connection()

        try:
            connection.execute(
                """
                DELETE FROM projects
                WHERE project_id = ?
                """,
                (project_id,),
            )
            connection.commit()
        finally:
            connection.close()
