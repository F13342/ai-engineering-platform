from ai_platform.database.connection import get_connection


class ConversationRepository:
    """Provide CRUD operations for conversations."""

    def create_conversation(
        self,
        user_id: int,
        title: str,
        project_id: int | None = None,
    ) -> int:
        """Create a conversation and return its ID."""
        connection = get_connection()

        try:
            cursor = connection.execute(
                """
                INSERT INTO conversations (user_id, project_id, title)
                VALUES (?, ?, ?)
                """,
                (user_id, project_id, title),
            )
            connection.commit()
            return int(cursor.lastrowid)
        finally:
            connection.close()

    def get_conversation(
        self,
        conversation_id: int,
    ) -> dict[str, object] | None:
        """Return a conversation by ID."""
        connection = get_connection()

        try:
            row = connection.execute(
                """
                SELECT
                    conversation_id,
                    user_id,
                    project_id,
                    title,
                    created_at
                FROM conversations
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()

            if row is None:
                return None

            return {
                "conversation_id": row[0],
                "user_id": row[1],
                "project_id": row[2],
                "title": row[3],
                "created_at": row[4],
            }
        finally:
            connection.close()

    def get_conversations_by_user(
        self,
        user_id: int,
    ) -> list[dict[str, object]]:
        """Return all conversations belonging to a user."""
        connection = get_connection()

        try:
            rows = connection.execute(
                """
                SELECT
                    conversation_id,
                    user_id,
                    project_id,
                    title,
                    created_at
                FROM conversations
                WHERE user_id = ?
                ORDER BY conversation_id
                """,
                (user_id,),
            ).fetchall()

            return [
                {
                    "conversation_id": row[0],
                    "user_id": row[1],
                    "project_id": row[2],
                    "title": row[3],
                    "created_at": row[4],
                }
                for row in rows
            ]
        finally:
            connection.close()

    def update_conversation(
        self,
        conversation_id: int,
        title: str,
    ) -> None:
        """Update a conversation title."""
        connection = get_connection()

        try:
            connection.execute(
                """
                UPDATE conversations
                SET title = ?
                WHERE conversation_id = ?
                """,
                (title, conversation_id),
            )
            connection.commit()
        finally:
            connection.close()

    def delete_conversation(self, conversation_id: int) -> None:
        """Delete a conversation by ID."""
        connection = get_connection()

        try:
            connection.execute(
                """
                DELETE FROM conversations
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            )
            connection.commit()
        finally:
            connection.close()
