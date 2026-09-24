from ai_platform.database.queries.running import (
    get_conversation_running_metrics,
)
from ai_platform.database.repositories.conversation_repository import (
    ConversationRepository,
)
from ai_platform.database.repositories.message_repository import MessageRepository
from ai_platform.database.repositories.user_repository import UserRepository
from ai_platform.database.schema import initialize_schema


def test_running_metrics() -> None:
    """Test running totals and running averages."""
    initialize_schema()

    user_repository = UserRepository()
    conversation_repository = ConversationRepository()
    message_repository = MessageRepository()

    user_id = user_repository.create_user(
        username="Running User",
        email="running.user@example.com",
    )

    conversation_id = conversation_repository.create_conversation(
        user_id=user_id,
        title="Running Metrics Test",
    )

    message_repository.create_message(
        conversation_id=conversation_id,
        role="user",
        content="Hello",
    )

    message_repository.create_message(
        conversation_id=conversation_id,
        role="assistant",
        content="Hello there!",
    )

    message_repository.create_message(
        conversation_id=conversation_id,
        role="user",
        content="Explain RAG.",
    )

    results = get_conversation_running_metrics(conversation_id)

    assert len(results) == 3

    assert results[0]["running_message_count"] == 1
    assert results[1]["running_message_count"] == 2
    assert results[2]["running_message_count"] == 3

    assert results[0]["running_average_length"] == 5.0

    assert results[1]["running_average_length"] == 8.5

    assert results[2]["running_average_length"] == (
        5 + 12 + 12
    ) / 3

    conversation_repository.delete_conversation(conversation_id)
    user_repository.delete_user(user_id)
