from ai_platform.database.connection import get_connection
from ai_platform.database.schema import initialize_schema


def test_database_schema() -> None:
    """Test that all core AI Platform tables are created."""
    initialize_schema()

    connection = get_connection()

    tables = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        ORDER BY name
        """
    ).fetchall()

    connection.close()

    table_names = {row[0] for row in tables}

    assert {
        "users",
        "projects",
        "conversations",
        "messages",
        "documents",
        "models",
    }.issubset(table_names)
