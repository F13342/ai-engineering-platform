import sqlite3
from pathlib import Path


DATABASE_PATH = Path("database/ai_platform.db")


def get_connection(
    database_path: Path = DATABASE_PATH,
) -> sqlite3.Connection:
    """Create and return a SQLite database connection."""
    database_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")

    return connection
