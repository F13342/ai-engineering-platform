from ai_platform.database.connection import get_connection


class UserRepository:
    """Provide CRUD operations for users."""

    def create_user(self, username: str, email: str) -> int:
        """Create a user and return its ID."""
        connection = get_connection()

        try:
            cursor = connection.execute(
                """
                INSERT INTO users (username, email)
                VALUES (?, ?)
                """,
                (username, email),
            )
            connection.commit()
            return int(cursor.lastrowid)
        finally:
            connection.close()

    def get_user(self, user_id: int) -> dict[str, object] | None:
        """Return a user by ID."""
        connection = get_connection()

        try:
            row = connection.execute(
                """
                SELECT user_id, username, email, created_at
                FROM users
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()

            if row is None:
                return None

            return {
                "user_id": row[0],
                "username": row[1],
                "email": row[2],
                "created_at": row[3],
            }
        finally:
            connection.close()

    def update_user(
        self,
        user_id: int,
        username: str,
        email: str,
    ) -> None:
        """Update an existing user."""
        connection = get_connection()

        try:
            connection.execute(
                """
                UPDATE users
                SET username = ?, email = ?
                WHERE user_id = ?
                """,
                (username, email, user_id),
            )
            connection.commit()
        finally:
            connection.close()

    def delete_user(self, user_id: int) -> None:
        """Delete a user by ID."""
        connection = get_connection()

        try:
            connection.execute(
                """
                DELETE FROM users
                WHERE user_id = ?
                """,
                (user_id,),
            )
            connection.commit()
        finally:
            connection.close()
