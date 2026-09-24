from ai_platform.database.connection import get_connection


def get_message_sequence_analysis() -> list[dict[str, object]]:
    """Compare each message with the previous and next message."""
    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                message_id,
                conversation_id,
                role,
                content,
                LAG(role) OVER (
                    PARTITION BY conversation_id
                    ORDER BY message_id
                ) AS previous_role,
                LEAD(role) OVER (
                    PARTITION BY conversation_id
                    ORDER BY message_id
                ) AS next_role
            FROM messages
            ORDER BY conversation_id, message_id
            """
        ).fetchall()

        return [
            {
                "message_id": row[0],
                "conversation_id": row[1],
                "role": row[2],
                "content": row[3],
                "previous_role": row[4],
                "next_role": row[5],
            }
            for row in rows
        ]
    finally:
        connection.close()
