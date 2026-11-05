from ai_platform.database.connection import get_connection


class MessageRepository:
    """Provide CRUD operations for messages."""

    def create_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
    ) -> int:
        """Create a message and return its ID."""
        connection = get_connection()

        try:
            cursor = connection.execute(
                """
                INSERT INTO messages (conversation_id, role, content)
                VALUES (?, ?, ?)
                """,
                (conversation_id, role, content),
            )
            connection.commit()
            return int(cursor.lastrowid)
        finally:
            connection.close()

    def create_message_pair(
        self,
        user_id: int,
        conversation_id: int,
        user_content: str,
        assistant_content: str,
    ) -> tuple[int, int]:
        """Atomically save a user message and its assistant reply."""
        connection = get_connection()

        try:
            connection.execute("BEGIN IMMEDIATE")

            owned = connection.execute(
                """
                SELECT 1
                FROM conversations
                WHERE conversation_id = ? AND user_id = ?
                """,
                (conversation_id, user_id),
            ).fetchone()

            if owned is None:
                raise LookupError("Conversation not found.")

            user_cursor = connection.execute(
                """
                INSERT INTO messages (conversation_id, role, content)
                VALUES (?, 'user', ?)
                """,
                (conversation_id, user_content),
            )

            assistant_cursor = connection.execute(
                """
                INSERT INTO messages (conversation_id, role, content)
                VALUES (?, 'assistant', ?)
                """,
                (conversation_id, assistant_content),
            )

            message_ids = (
                int(user_cursor.lastrowid),
                int(assistant_cursor.lastrowid),
            )
            connection.commit()
            return message_ids

        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def get_message(
        self,
        message_id: int,
    ) -> dict[str, object] | None:
        """Return a message by ID."""
        connection = get_connection()

        try:
            row = connection.execute(
                """
                SELECT
                    message_id,
                    conversation_id,
                    role,
                    content,
                    created_at
                FROM messages
                WHERE message_id = ?
                """,
                (message_id,),
            ).fetchone()

            if row is None:
                return None

            return {
                "message_id": row[0],
                "conversation_id": row[1],
                "role": row[2],
                "content": row[3],
                "created_at": row[4],
            }
        finally:
            connection.close()

    def get_messages_by_conversation(
        self,
        conversation_id: int,
    ) -> list[dict[str, object]]:
        """Return messages ordered by creation sequence."""
        connection = get_connection()

        try:
            rows = connection.execute(
                """
                SELECT
                    message_id,
                    conversation_id,
                    role,
                    content,
                    created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY message_id
                """,
                (conversation_id,),
            ).fetchall()

            return [
                {
                    "message_id": row[0],
                    "conversation_id": row[1],
                    "role": row[2],
                    "content": row[3],
                    "created_at": row[4],
                }
                for row in rows
            ]
        finally:
            connection.close()

    def update_message(
        self,
        message_id: int,
        content: str,
    ) -> None:
        """Update a message content."""
        connection = get_connection()

        try:
            connection.execute(
                """
                UPDATE messages
                SET content = ?
                WHERE message_id = ?
                """,
                (content, message_id),
            )
            connection.commit()
        finally:
            connection.close()

    def delete_message(self, message_id: int) -> None:
        """Delete a message by ID."""
        connection = get_connection()

        try:
            connection.execute(
                """
                DELETE FROM messages
                WHERE message_id = ?
                """,
                (message_id,),
            )
            connection.commit()
        finally:
            connection.close()
