import asyncio

import httpx
import pytest

from ai_platform.api.app import create_app


class FakeConversationService:
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
            "created_at": "2026-11-05 12:00:00",
        }
        return conversation_id

    def get_conversation(self, user_id, conversation_id):
        record = self.records.get(conversation_id)
        if record is None or record["user_id"] != user_id:
            raise LookupError("Conversation not found.")
        return {
            "conversation": record,
            "messages": [
                {
                    "message_id": 1,
                    "role": "user",
                    "content": "Hello",
                    "created_at": "2026-11-05 12:01:00",
                }
            ],
        }

    def list_conversations(self, user_id):
        return [
            record for record in self.records.values()
            if record["user_id"] == user_id
        ]

    def delete_conversation(self, user_id, conversation_id):
        self.get_conversation(user_id, conversation_id)
        del self.records[conversation_id]


@pytest.fixture(autouse=True)
def configure_api_key(monkeypatch):
    monkeypatch.setenv("AI_PLATFORM_API_KEY", "test-secret")
    monkeypatch.setenv("AI_PLATFORM_LOCAL_USER_ID", "920")


def make_request(app, method, path, **kwargs):
    async def send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def auth_headers():
    return {"X-API-Key": "test-secret"}


def test_create_conversation():
    service = FakeConversationService()
    app = create_app(conversation_service=service)

    response = make_request(
        app,
        "POST",
        "/v1/conversations",
        json={"title": "My persistent chat"},
        headers=auth_headers(),
    )

    assert response.status_code == 201
    assert response.json()["conversation_id"] == 1
    assert response.json()["title"] == "My persistent chat"
    assert service.records[1]["user_id"] == 920


def test_list_conversations():
    service = FakeConversationService()
    service.create_conversation(920, "First")
    service.create_conversation(920, "Second")
    service.create_conversation(999, "Other user's chat")

    app = create_app(conversation_service=service)

    response = make_request(
        app,
        "GET",
        "/v1/conversations",
        headers=auth_headers(),
    )

    assert response.status_code == 200
    assert [item["title"] for item in response.json()] == ["First", "Second"]


def test_get_conversation_includes_messages():
    service = FakeConversationService()
    conversation_id = service.create_conversation(920, "History")
    app = create_app(conversation_service=service)

    response = make_request(
        app,
        "GET",
        f"/v1/conversations/{conversation_id}",
        headers=auth_headers(),
    )

    assert response.status_code == 200
    assert response.json()["messages"][0]["content"] == "Hello"


def test_get_other_users_conversation_returns_404():
    service = FakeConversationService()
    conversation_id = service.create_conversation(999, "Private")
    app = create_app(conversation_service=service)

    response = make_request(
        app,
        "GET",
        f"/v1/conversations/{conversation_id}",
        headers=auth_headers(),
    )

    assert response.status_code == 404


def test_delete_conversation():
    service = FakeConversationService()
    conversation_id = service.create_conversation(920, "To delete")
    app = create_app(conversation_service=service)

    response = make_request(
        app,
        "DELETE",
        f"/v1/conversations/{conversation_id}",
        headers=auth_headers(),
    )

    assert response.status_code == 200
    assert response.json()["status"] == "deleted"
    assert conversation_id not in service.records


def test_conversation_api_requires_api_key():
    app = create_app(conversation_service=FakeConversationService())

    response = make_request(
        app,
        "GET",
        "/v1/conversations",
    )

    assert response.status_code == 401
