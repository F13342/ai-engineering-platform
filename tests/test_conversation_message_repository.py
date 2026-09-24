from ai_platform.database.repositories.conversation_repository import (
    ConversationRepository,
)
from ai_platform.database.repositories.message_repository import MessageRepository
from ai_platform.database.repositories.user_repository import UserRepository
from ai_platform.database.schema import initialize_schema


def test_conversation_and_message_crud() -> None:
    """Test conversations and messages together."""
    initialize_schema()

    user_repository = UserRepository()
    conversation_repository = ConversationRepository()
    message_repository = MessageRepository()

    user_id = user_repository.create_user(
        username="Chat User",
        email="chat.user@example.com",
    )

    conversation_id = conversation_repository.create_conversation(
        user_id=user_id,
        title="AI Assistant Session",
    )

    conversation = conversation_repository.get_conversation(conversation_id)

    assert conversation is not None
    assert conversation["user_id"] == user_id
    assert conversation["title"] == "AI Assistant Session"

    message_id_1 = message_repository.create_message(
        conversation_id=conversation_id,
        role="user",
        content="Explain machine learning.",
    )

    message_id_2 = message_repository.create_message(
        conversation_id=conversation_id,
        role="assistant",
        content="Machine learning is a method of learning patterns from data.",
    )

    messages = message_repository.get_messages_by_conversation(
        conversation_id
    )

    assert len(messages) == 2
    assert messages[0]["message_id"] == message_id_1
    assert messages[0]["role"] == "user"
    assert messages[1]["message_id"] == message_id_2
    assert messages[1]["role"] == "assistant"

    message_repository.update_message(
        message_id=message_id_1,
        content="Explain deep learning.",
    )

    updated_message = message_repository.get_message(message_id_1)

    assert updated_message is not None
    assert updated_message["content"] == "Explain deep learning."

    conversation_repository.update_conversation(
        conversation_id=conversation_id,
        title="AI Assistant Session Updated",
    )

    updated_conversation = conversation_repository.get_conversation(
        conversation_id
    )

    assert updated_conversation is not None
    assert updated_conversation["title"] == "AI Assistant Session Updated"

    message_repository.delete_message(message_id_2)

    remaining_messages = message_repository.get_messages_by_conversation(
        conversation_id
    )

    assert len(remaining_messages) == 1

    conversation_repository.delete_conversation(conversation_id)

    assert (
        conversation_repository.get_conversation(conversation_id)
        is None
    )

    user_repository.delete_user(user_id)
