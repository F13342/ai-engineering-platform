from ai_platform.database.connection import get_connection


def get_conversation_running_metrics(
    conversation_id: int,
) -> list[dict[str, object]]:
    """Return running message count and average message length."""
    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                message_id,
                role,
                LENGTH(content) AS content_length,
                SUM(1) OVER (
                    ORDER BY message_id
                    ROWS BETWEEN UNBOUNDED PRECEDING
                    AND CURRENT ROW
                ) AS running_message_count,
                AVG(LENGTH(content)) OVER (
                    ORDER BY message_id
                    ROWS BETWEEN UNBOUNDED PRECEDING
                    AND CURRENT ROW
                ) AS running_average_length
            FROM messages
            WHERE conversation_id = ?
            ORDER BY message_id
            """,
            (conversation_id,),
        ).fetchall()

        return [
            {
                "message_id": row[0],
                "role": row[1],
                "content_length": row[2],
                "running_message_count": row[3],
                "running_average_length": row[4],
            }
            for row in rows
        ]
    finally:
        connection.close()
