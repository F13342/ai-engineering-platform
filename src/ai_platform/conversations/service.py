"""Persistent conversation management service."""

from ai_platform.database.repositories.conversation_repository import (
    ConversationRepository,
)
from ai_platform.database.repositories.message_repository import (
    MessageRepository,
)
from ai_platform.database.repositories.user_repository import (
    UserRepository,
)


class ConversationService:
    """Manage conversations with user-ownership checks."""

    def __init__(
        self,
        conversation_repository=None,
        message_repository=None,
        user_repository=None,
    ) -> None:
        self.conversations = (
            conversation_repository or ConversationRepository()
        )
        self.messages = message_repository or MessageRepository()
        self.users = user_repository or UserRepository()

    def _require_user(self, user_id: int) -> None:
        if user_id <= 0 or self.users.get_user(user_id) is None:
            raise LookupError("User not found.")

    def _require_owned_conversation(
        self,
        user_id: int,
        conversation_id: int,
    ) -> dict[str, object]:
        self._require_user(user_id)

        conversation = self.conversations.get_conversation(
            conversation_id
        )

        if (
            conversation is None
            or conversation["user_id"] != user_id
        ):
            raise LookupError("Conversation not found.")

        return conversation

    def create_conversation(
        self,
        user_id: int,
        title: str = "New conversation",
        project_id: int | None = None,
    ) -> int:
        self._require_user(user_id)

        title = title.strip()
        if not title:
            raise ValueError("Conversation title cannot be blank.")

        return self.conversations.create_conversation(
            user_id=user_id,
            title=title,
            project_id=project_id,
        )

    def list_conversations(
        self,
        user_id: int,
    ) -> list[dict[str, object]]:
        self._require_user(user_id)
        return self.conversations.get_conversations_by_user(user_id)

    def get_conversation(
        self,
        user_id: int,
        conversation_id: int,
    ) -> dict[str, object]:
        conversation = self._require_owned_conversation(
            user_id,
            conversation_id,
        )

        messages = self.messages.get_messages_by_conversation(
            conversation_id
        )

        return {
            "conversation": conversation,
            "messages": messages,
        }

    def add_message(
        self,
        user_id: int,
        conversation_id: int,
        role: str,
        content: str,
    ) -> int:
        self._require_owned_conversation(user_id, conversation_id)

        if role not in {"user", "assistant"}:
            raise ValueError("Unsupported conversation message role.")

        content = content.strip()
        if not content:
            raise ValueError("Message content cannot be blank.")

        return self.messages.create_message(
            conversation_id=conversation_id,
            role=role,
            content=content,
        )

    def add_exchange(
        self,
        user_id: int,
        conversation_id: int,
        user_content: str,
        assistant_content: str,
    ) -> tuple[int, int]:
        """Save both sides of a chat exchange in one transaction."""
        self._require_owned_conversation(user_id, conversation_id)

        user_content = user_content.strip()
        assistant_content = assistant_content.strip()

        if not user_content:
            raise ValueError("User message cannot be blank.")

        if not assistant_content:
            raise ValueError("Assistant message cannot be blank.")

        return self.messages.create_message_pair(
            user_id=user_id,
            conversation_id=conversation_id,
            user_content=user_content,
            assistant_content=assistant_content,
        )

    def get_recent_history(
        self,
        user_id: int,
        conversation_id: int,
        limit: int = 6,
    ) -> list[dict[str, str]]:
        if limit < 1 or limit > 50:
            raise ValueError("History limit must be between 1 and 50.")

        result = self.get_conversation(user_id, conversation_id)

        return [
            {
                "role": str(message["role"]),
                "content": str(message["content"]),
            }
            for message in result["messages"][-limit:]
        ]

    def delete_conversation(
        self,
        user_id: int,
        conversation_id: int,
    ) -> None:
        self._require_owned_conversation(user_id, conversation_id)
        self.conversations.delete_conversation(conversation_id)
