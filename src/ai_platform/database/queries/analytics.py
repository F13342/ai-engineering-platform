from ai_platform.database.connection import get_connection


def get_user_project_statistics() -> list[dict[str, object]]:
    """Return project statistics for each user."""
    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                u.user_id,
                u.username,
                COUNT(p.project_id) AS project_count
            FROM users AS u
            LEFT JOIN projects AS p
                ON u.user_id = p.user_id
            GROUP BY
                u.user_id,
                u.username
            ORDER BY project_count DESC, u.user_id
            """
        ).fetchall()

        return [
            {
                "user_id": row[0],
                "username": row[1],
                "project_count": row[2],
            }
            for row in rows
        ]
    finally:
        connection.close()


def get_users_with_minimum_projects(
    minimum_projects: int,
) -> list[dict[str, object]]:
    """Return users whose project count meets a minimum."""
    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                u.user_id,
                u.username,
                COUNT(p.project_id) AS project_count
            FROM users AS u
            LEFT JOIN projects AS p
                ON u.user_id = p.user_id
            GROUP BY
                u.user_id,
                u.username
            HAVING COUNT(p.project_id) >= ?
            ORDER BY project_count DESC
            """,
            (minimum_projects,),
        ).fetchall()

        return [
            {
                "user_id": row[0],
                "username": row[1],
                "project_count": row[2],
            }
            for row in rows
        ]
    finally:
        connection.close()
