from ai_platform.database.connection import get_connection


def get_user_activity_ranking() -> list[dict[str, object]]:
    """Rank users by the number of projects they own."""
    connection = get_connection()

    try:
        rows = connection.execute(
            """
            WITH user_metrics AS (
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
            )
            SELECT
                user_id,
                username,
                project_count,
                ROW_NUMBER() OVER (
                    ORDER BY project_count DESC, user_id
                ) AS row_number,
                RANK() OVER (
                    ORDER BY project_count DESC
                ) AS activity_rank
            FROM user_metrics
            ORDER BY activity_rank, user_id
            """
        ).fetchall()

        return [
            {
                "user_id": row[0],
                "username": row[1],
                "project_count": row[2],
                "row_number": row[3],
                "activity_rank": row[4],
            }
            for row in rows
        ]
    finally:
        connection.close()
