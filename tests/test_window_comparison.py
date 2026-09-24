from ai_platform.database.queries.window_comparison import (
    get_message_sequence_analysis,
)
from ai_platform.database.repositories.conversation_repository import (
    ConversationRepository,
)
from ai_platform.database.repositories.message_repository import MessageRepository
from ai_platform.database.repositories.user_repository import UserRepository
from ai_platform.database.schema import initialize_schema


def test_lag_and_lead_queries() -> None:
    """Test LAG and LEAD over conversation messages."""
    initialize_schema()

    user_repository = UserRepository()
    conversation_repository = ConversationRepository()
    message_repository = MessageRepository()

    user_id = user_repository.create_user(
        username="Window User",
        email="window.user@example.com",
    )

    conversation_id = conversation_repository.create_conversation(
        user_id=user_id,
        title="Window Function Test",
    )

    message_1 = message_repository.create_message(
        conversation_id=conversation_id,
        role="user",
        content="Hello AI.",
    )

    message_2 = message_repository.create_message(
        conversation_id=conversation_id,
        role="assistant",
        content="Hello! How can I help?",
    )

    message_3 = message_repository.create_message(
        conversation_id=conversation_id,
        role="user",
        content="Explain RAG.",
    )

    results = get_message_sequence_analysis()

    result_1 = next(
        item for item in results if item["message_id"] == message_1
    )
    result_2 = next(
        item for item in results if item["message_id"] == message_2
    )
    result_3 = next(
        item for item in results if item["message_id"] == message_3
    )

    assert result_1["previous_role"] is None
    assert result_1["next_role"] == "assistant"

    assert result_2["previous_role"] == "user"
    assert result_2["next_role"] == "user"

    assert result_3["previous_role"] == "assistant"
    assert result_3["next_role"] is None

    conversation_repository.delete_conversation(conversation_id)
    user_repository.delete_user(user_id)
