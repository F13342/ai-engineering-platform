from importlib import import_module

from ai_platform.database.connection import get_connection
from ai_platform.database.analytics.user_activity import (
    build_user_analytical_dataset,
)
from ai_platform.database.repositories.conversation_repository import (
    ConversationRepository,
)
from ai_platform.database.repositories.document_repository import (
    DocumentRepository,
)
from ai_platform.database.repositories.message_repository import (
    MessageRepository,
)
from ai_platform.database.repositories.project_repository import (
    ProjectRepository,
)
from ai_platform.database.repositories.user_repository import (
    UserRepository,
)
from ai_platform.database.schema import initialize_schema


def test_user_analytical_dataset(tmp_path, monkeypatch) -> None:
    """Test the complete user analytical dataset using isolated SQLite."""

    database_path = tmp_path / "analytical_dataset.db"

    def temporary_connection():
        return get_connection(database_path)

    database_modules = [
        "ai_platform.database.schema",
        "ai_platform.database.repositories.user_repository",
        "ai_platform.database.repositories.project_repository",
        "ai_platform.database.repositories.conversation_repository",
        "ai_platform.database.repositories.message_repository",
        "ai_platform.database.repositories.document_repository",
        "ai_platform.database.analytics.user_activity",
    ]

    for module_name in database_modules:
        module = import_module(module_name)
        monkeypatch.setattr(
            module,
            "get_connection",
            temporary_connection,
        )

    initialize_schema()

    user_repository = UserRepository()
    project_repository = ProjectRepository()
    conversation_repository = ConversationRepository()
    message_repository = MessageRepository()
    document_repository = DocumentRepository()

    user_id = user_repository.create_user(
        username="Analytical User",
        email="analytical.user@example.com",
    )

    project_id = project_repository.create_project(
        user_id=user_id,
        name="Analytics Project",
        description="Project for analytical dataset testing.",
    )

    conversation_id = conversation_repository.create_conversation(
        user_id=user_id,
        project_id=project_id,
        title="Analytics Conversation",
    )

    message_repository.create_message(
        conversation_id=conversation_id,
        role="user",
        content="Build an AI system.",
    )

    message_repository.create_message(
        conversation_id=conversation_id,
        role="assistant",
        content="The system will use data, models, and agents.",
    )

    document_repository.create_document(
        project_id=project_id,
        filename="architecture.pdf",
        file_type="pdf",
        status="processed",
    )

    results = build_user_analytical_dataset()

    result = next(
        item
        for item in results
        if item["user_id"] == user_id
    )

    assert result["project_count"] == 1
    assert result["conversation_count"] == 1
    assert result["message_count"] == 2
    assert result["document_count"] == 1
    assert result["activity_total"] == 5
    assert result["activity_rank"] == 1
    assert result["average_activity"] == 5.0
    assert result["difference_from_average"] == 0.0
    assert result["activity_level"] == "Medium"
