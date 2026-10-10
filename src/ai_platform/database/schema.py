import sqlite3

from ai_platform.database.connection import get_connection


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS projects (
    project_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id)
        REFERENCES users (user_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    project_id INTEGER,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id)
        REFERENCES users (user_id)
        ON DELETE CASCADE,
    FOREIGN KEY (project_id)
        REFERENCES projects (project_id)
        ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (conversation_id)
        REFERENCES conversations (conversation_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS documents (
    document_id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id)
        REFERENCES projects (project_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS models (
    model_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    version TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT 'local',
    task_type TEXT NOT NULL DEFAULT 'unknown',
    framework TEXT NOT NULL DEFAULT 'unknown',
    artifact_uri TEXT,
    status TEXT NOT NULL DEFAULT 'registered',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    parameters_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


MODEL_REGISTRY_COLUMNS = {
    "provider": "TEXT NOT NULL DEFAULT 'local'",
    "task_type": "TEXT NOT NULL DEFAULT 'unknown'",
    "framework": "TEXT NOT NULL DEFAULT 'unknown'",
    "artifact_uri": "TEXT",
    "status": "TEXT NOT NULL DEFAULT 'registered'",
    "metrics_json": "TEXT NOT NULL DEFAULT '{}'",
    "parameters_json": "TEXT NOT NULL DEFAULT '{}'",
}


def _migrate_model_registry(connection: sqlite3.Connection) -> None:
    """Add model registry fields to existing databases without losing data."""
    existing_columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(models)")
    }

    for column, definition in MODEL_REGISTRY_COLUMNS.items():
        if column not in existing_columns:
            connection.execute(
                f"ALTER TABLE models ADD COLUMN {column} {definition}"
            )


def initialize_schema() -> None:
    """Create the AI Platform database schema."""
    connection = get_connection()

    try:
        connection.executescript(SCHEMA_SQL)
        _migrate_model_registry(connection)
        connection.commit()
    finally:
        connection.close()
