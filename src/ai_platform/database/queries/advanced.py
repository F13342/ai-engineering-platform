from ai_platform.database.connection import get_connection


def get_user_activity_analysis() -> list[dict[str, object]]:
    """Build user activity metrics with CTE and CASE."""
    connection = get_connection()

    try:
        rows = connection.execute(
            """
            WITH user_metrics AS (
                SELECT
                    u.user_id,
                    u.username,
                    COUNT(DISTINCT p.project_id) AS project_count,
                    COUNT(DISTINCT c.conversation_id) AS conversation_count,
                    COUNT(DISTINCT d.document_id) AS document_count
                FROM users AS u
                LEFT JOIN projects AS p
                    ON u.user_id = p.user_id
                LEFT JOIN conversations AS c
                    ON u.user_id = c.user_id
                LEFT JOIN documents AS d
                    ON p.project_id = d.project_id
                GROUP BY
                    u.user_id,
                    u.username
            )
            SELECT
                user_id,
                username,
                project_count,
                conversation_count,
                document_count,
                CASE
                    WHEN project_count >= 3
                         OR conversation_count >= 5
                    THEN 'High'

                    WHEN project_count >= 1
                         OR conversation_count >= 1
                    THEN 'Medium'

                    ELSE 'Low'
                END AS activity_level
            FROM user_metrics
            ORDER BY project_count DESC, conversation_count DESC
            """
        ).fetchall()

        return [
            {
                "user_id": row[0],
                "username": row[1],
                "project_count": row[2],
                "conversation_count": row[3],
                "document_count": row[4],
                "activity_level": row[5],
            }
            for row in rows
        ]
    finally:
        connection.close()
