from ai_platform.database.connection import get_connection


def build_user_analytical_dataset() -> list[dict[str, object]]:
    """Build one analytical row per AI Platform user."""
    connection = get_connection()

    try:
        rows = connection.execute(
            """
            WITH project_metrics AS (
                SELECT
                    user_id,
                    COUNT(*) AS project_count
                FROM projects
                GROUP BY user_id
            ),
            conversation_metrics AS (
                SELECT
                    user_id,
                    COUNT(*) AS conversation_count
                FROM conversations
                GROUP BY user_id
            ),
            message_metrics AS (
                SELECT
                    c.user_id,
                    COUNT(m.message_id) AS message_count
                FROM conversations AS c
                LEFT JOIN messages AS m
                    ON c.conversation_id = m.conversation_id
                GROUP BY c.user_id
            ),
            document_metrics AS (
                SELECT
                    p.user_id,
                    COUNT(d.document_id) AS document_count
                FROM projects AS p
                LEFT JOIN documents AS d
                    ON p.project_id = d.project_id
                GROUP BY p.user_id
            ),
            user_metrics AS (
                SELECT
                    u.user_id,
                    u.username,
                    COALESCE(pm.project_count, 0) AS project_count,
                    COALESCE(cm.conversation_count, 0) AS conversation_count,
                    COALESCE(mm.message_count, 0) AS message_count,
                    COALESCE(dm.document_count, 0) AS document_count
                FROM users AS u
                LEFT JOIN project_metrics AS pm
                    ON u.user_id = pm.user_id
                LEFT JOIN conversation_metrics AS cm
                    ON u.user_id = cm.user_id
                LEFT JOIN message_metrics AS mm
                    ON u.user_id = mm.user_id
                LEFT JOIN document_metrics AS dm
                    ON u.user_id = dm.user_id
            ),
            activity_metrics AS (
                SELECT
                    *,
                    (
                        project_count
                        + conversation_count
                        + message_count
                        + document_count
                    ) AS activity_total
                FROM user_metrics
            )
            SELECT
                user_id,
                username,
                project_count,
                conversation_count,
                message_count,
                document_count,
                activity_total,
                RANK() OVER (
                    ORDER BY activity_total DESC
                ) AS activity_rank,
                AVG(activity_total) OVER () AS average_activity,
                activity_total - AVG(activity_total) OVER ()
                    AS difference_from_average,
                CASE
                    WHEN activity_total >= 10 THEN 'High'
                    WHEN activity_total >= 3 THEN 'Medium'
                    ELSE 'Low'
                END AS activity_level
            FROM activity_metrics
            ORDER BY activity_rank, user_id
            """
        ).fetchall()

        return [
            {
                "user_id": row[0],
                "username": row[1],
                "project_count": row[2],
                "conversation_count": row[3],
                "message_count": row[4],
                "document_count": row[5],
                "activity_total": row[6],
                "activity_rank": row[7],
                "average_activity": row[8],
                "difference_from_average": row[9],
                "activity_level": row[10],
            }
            for row in rows
        ]
    finally:
        connection.close()
