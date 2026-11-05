import sqlite3

from ai_platform.conversations.service import ConversationService


def test_conversation_persists_across_service_instances(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "conversation_test.db"

    def connect():
        connection = sqlite3.connect(database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    with connect() as connection:
        connection.executescript("""
            CREATE TABLE users (
                user_id INTEGER PRIMARY KEY,
                username TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE projects (
                project_id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            CREATE TABLE conversations (
                conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                project_id INTEGER,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (project_id) REFERENCES projects(project_id)
            );

            CREATE TABLE messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(conversation_id)
                    ON DELETE CASCADE
            );

            INSERT INTO users (user_id, username, email)
            VALUES (920, 'local-owner', 'owner@ai-platform.local');
        """)

    monkeypatch.setattr(
        "ai_platform.database.repositories.conversation_repository.get_connection",
        connect,
    )
    monkeypatch.setattr(
        "ai_platform.database.repositories.message_repository.get_connection",
        connect,
    )
    monkeypatch.setattr(
        "ai_platform.database.repositories.user_repository.get_connection",
        connect,
    )

    # First service instance: create and save messages.
    first_service = ConversationService()

    conversation_id = first_service.create_conversation(
        user_id=920,
        title="Persistent memory test",
    )

    first_service.add_exchange(
        920,
        conversation_id,
        "My project is Nibras.",
        "I will remember Nibras.",
    )

    # Simulate a failure on the assistant insert. The user insert
    # from the same exchange must be rolled back as well.
    with connect() as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_assistant_message
            BEFORE INSERT ON messages
            WHEN NEW.role = 'assistant'
            BEGIN
                SELECT RAISE(ABORT, 'simulated assistant insert failure');
            END;
            """
        )

    try:
        first_service.add_exchange(
            920,
            conversation_id,
            "This message must be rolled back.",
            "This assistant reply will fail.",
        )
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("Expected the simulated insert failure.")

    with connect() as connection:
        stored_count = connection.execute(
            """
            SELECT COUNT(*)
            FROM messages
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()[0]

    assert stored_count == 2

    with connect() as connection:
        connection.execute("DROP TRIGGER reject_assistant_message")

    # Second service instance: restore data from SQLite.
    restored_service = ConversationService()

    result = restored_service.get_conversation(
        920,
        conversation_id,
    )

    assert result["conversation"]["title"] == "Persistent memory test"
    assert len(result["messages"]) == 2
    assert result["messages"][0]["content"] == "My project is Nibras."
    assert result["messages"][1]["content"] == "I will remember Nibras."

    history = restored_service.get_recent_history(
        920,
        conversation_id,
    )

    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_deleting_conversation_cascades_to_messages(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "cascade_test.db"

    def connect():
        connection = sqlite3.connect(database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    with connect() as connection:
        connection.executescript("""
            CREATE TABLE users (
                user_id INTEGER PRIMARY KEY,
                username TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE projects (
                project_id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            CREATE TABLE conversations (
                conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                project_id INTEGER,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (project_id) REFERENCES projects(project_id)
            );

            CREATE TABLE messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(conversation_id)
                    ON DELETE CASCADE
            );

            INSERT INTO users (user_id, username, email) VALUES (
                920, 'local-owner', 'owner@ai-platform.local'
            );
        """)

    monkeypatch.setattr(
        "ai_platform.database.repositories.conversation_repository.get_connection",
        connect,
    )
    monkeypatch.setattr(
        "ai_platform.database.repositories.message_repository.get_connection",
        connect,
    )
    monkeypatch.setattr(
        "ai_platform.database.repositories.user_repository.get_connection",
        connect,
    )

    service = ConversationService()
    conversation_id = service.create_conversation(920, "Delete test")
    service.add_message(920, conversation_id, "user", "Hello")

    service.delete_conversation(920, conversation_id)

    with connect() as connection:
        conversation_count = connection.execute(
            "SELECT COUNT(*) FROM conversations"
        ).fetchone()[0]

        message_count = connection.execute(
            "SELECT COUNT(*) FROM messages"
        ).fetchone()[0]

    assert conversation_count == 0
    assert message_count == 0
