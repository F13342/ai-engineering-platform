from ai_platform.database.connection import get_connection


def test_database_connection() -> None:
    """Test that the SQLite database connection works."""
    connection = get_connection()

    result = connection.execute("SELECT 1").fetchone()

    connection.close()

    assert result == (1,)
