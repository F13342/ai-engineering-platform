from ai_platform.conversations.service import ConversationService


class FakeUsers:
    def get_user(self, user_id):
        if user_id == 920:
            return {"user_id": 920}
        return None


class FakeConversations:
    def __init__(self):
        self.records = {}
        self.next_id = 1

    def create_conversation(self, user_id, title, project_id=None):
        conversation_id = self.next_id
        self.next_id += 1

        self.records[conversation_id] = {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "title": title,
            "project_id": project_id,
        }
        return conversation_id

    def get_conversation(self, conversation_id):
        return self.records.get(conversation_id)

    def get_conversations_by_user(self, user_id):
        return [
            item for item in self.records.values()
            if item["user_id"] == user_id
        ]

    def delete_conversation(self, conversation_id):
        self.records.pop(conversation_id, None)


class FakeMessages:
    def __init__(self):
        self.records = {}
        self.next_id = 1

    def create_message(self, conversation_id, role, content):
        message_id = self.next_id
        self.next_id += 1

        self.records.setdefault(conversation_id, []).append({
            "message_id": message_id,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
        })
        return message_id

    def create_message_pair(
        self,
        user_id,
        conversation_id,
        user_content,
        assistant_content,
    ):
        user_message_id = self.create_message(
            conversation_id, "user", user_content
        )
        assistant_message_id = self.create_message(
            conversation_id, "assistant", assistant_content
        )
        return user_message_id, assistant_message_id

    def get_messages_by_conversation(self, conversation_id):
        return list(self.records.get(conversation_id, []))


def make_service():
    conversations = FakeConversations()
    messages = FakeMessages()

    service = ConversationService(
        conversation_repository=conversations,
        message_repository=messages,
        user_repository=FakeUsers(),
    )

    return service, conversations, messages


def test_create_conversation_for_existing_user():
    service, _, _ = make_service()

    conversation_id = service.create_conversation(
        920,
        "My first conversation",
    )

    assert conversation_id == 1


def test_create_conversation_rejects_unknown_user():
    service, _, _ = make_service()

    try:
        service.create_conversation(54, "Invalid")
        assert False, "Expected user lookup failure"
    except LookupError:
        pass


def test_add_and_retrieve_messages():
    service, _, _ = make_service()

    conversation_id = service.create_conversation(920, "Test")

    service.add_message(920, conversation_id, "user", "Hello")
    service.add_message(920, conversation_id, "assistant", "Welcome")

    result = service.get_conversation(920, conversation_id)

    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][1]["content"] == "Welcome"


def test_add_exchange_saves_both_messages_in_order():
    service, _, messages = make_service()
    conversation_id = service.create_conversation(920, "Atomic exchange")

    user_message_id, assistant_message_id = service.add_exchange(
        920,
        conversation_id,
        "  Hello  ",
        "  Welcome  ",
    )

    stored = messages.get_messages_by_conversation(conversation_id)

    assert user_message_id == 1
    assert assistant_message_id == 2
    assert [message["role"] for message in stored] == [
        "user",
        "assistant",
    ]
    assert [message["content"] for message in stored] == [
        "Hello",
        "Welcome",
    ]


def test_history_returns_recent_messages_only():
    service, _, _ = make_service()
    conversation_id = service.create_conversation(920, "Test")

    for index in range(8):
        service.add_message(
            920,
            conversation_id,
            "user",
            f"Message {index}",
        )

    history = service.get_recent_history(
        920,
        conversation_id,
        limit=6,
    )

    assert len(history) == 6
    assert history[0]["content"] == "Message 2"
    assert history[-1]["content"] == "Message 7"


def test_user_cannot_read_another_users_conversation():
    service, conversations, _ = make_service()

    conversation_id = service.create_conversation(920, "Private")
    conversations.records[conversation_id]["user_id"] = 999

    try:
        service.get_conversation(920, conversation_id)
        assert False, "Expected ownership check to fail"
    except LookupError:
        pass


def test_user_cannot_delete_another_users_conversation():
    service, conversations, _ = make_service()

    conversation_id = service.create_conversation(920, "Private")
    conversations.records[conversation_id]["user_id"] = 999

    try:
        service.delete_conversation(920, conversation_id)
        assert False, "Expected ownership check to fail"
    except LookupError:
        pass

    assert conversation_id in conversations.records


def test_invalid_message_role_is_rejected():
    service, _, _ = make_service()
    conversation_id = service.create_conversation(920, "Test")

    try:
        service.add_message(920, conversation_id, "system", "Injected")
        assert False, "Expected role validation to fail"
    except ValueError:
        pass


def test_delete_owned_conversation():
    service, conversations, _ = make_service()
    conversation_id = service.create_conversation(920, "Test")

    service.delete_conversation(920, conversation_id)

    assert conversation_id not in conversations.records
