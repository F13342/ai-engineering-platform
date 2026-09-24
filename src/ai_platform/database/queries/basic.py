from ai_platform.database.connection import get_connection


def get_projects_with_users() -> list[dict[str, object]]:
    """Return projects with their owning users using an INNER JOIN."""
    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                p.project_id,
                p.name AS project_name,
                p.description,
                u.user_id,
                u.username,
                u.email
            FROM projects AS p
            INNER JOIN users AS u
                ON p.user_id = u.user_id
            ORDER BY p.project_id
            """
        ).fetchall()

        return [
            {
                "project_id": row[0],
                "project_name": row[1],
                "description": row[2],
                "user_id": row[3],
                "username": row[4],
                "email": row[5],
            }
            for row in rows
        ]
    finally:
        connection.close()


def get_user_projects(user_id: int) -> list[dict[str, object]]:
    """Return projects for a specific user."""
    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                project_id,
                name,
                description
            FROM projects
            WHERE user_id = ?
            ORDER BY project_id
            """,
            (user_id,),
        ).fetchall()

        return [
            {
                "project_id": row[0],
                "name": row[1],
                "description": row[2],
            }
            for row in rows
        ]
    finally:
        connection.close()


def search_projects(keyword: str) -> list[dict[str, object]]:
    """Search projects by name using WHERE and LIKE."""
    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                project_id,
                name,
                description
            FROM projects
            WHERE name LIKE ?
            ORDER BY name
            """,
            (f"%{keyword}%",),
        ).fetchall()

        return [
            {
                "project_id": row[0],
                "name": row[1],
                "description": row[2],
            }
            for row in rows
        ]
    finally:
        connection.close()
