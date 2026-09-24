from ai_platform.database.queries.advanced import get_user_activity_analysis
from ai_platform.database.repositories.conversation_repository import (
    ConversationRepository,
)
from ai_platform.database.repositories.project_repository import ProjectRepository
from ai_platform.database.repositories.user_repository import UserRepository
from ai_platform.database.schema import initialize_schema


def test_cte_and_case_analysis() -> None:
    """Test CTE-based metrics and CASE classification."""
    initialize_schema()

    user_repository = UserRepository()
    project_repository = ProjectRepository()
    conversation_repository = ConversationRepository()

    user_id = user_repository.create_user(
        username="Advanced SQL User",
        email="advanced.sql@example.com",
    )

    project_id = project_repository.create_project(
        user_id=user_id,
        name="AI RAG Project",
        description="Project for RAG development.",
    )

    conversation_repository.create_conversation(
        user_id=user_id,
        project_id=project_id,
        title="RAG Conversation",
    )

    analysis = get_user_activity_analysis()

    result = next(
        item
        for item in analysis
        if item["user_id"] == user_id
    )

    assert result["project_count"] == 1
    assert result["conversation_count"] == 1
    assert result["activity_level"] == "Medium"

    project_repository.delete_project(project_id)
    user_repository.delete_user(user_id)
