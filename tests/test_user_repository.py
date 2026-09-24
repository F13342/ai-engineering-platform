from ai_platform.database.schema import initialize_schema
from ai_platform.database.repositories.user_repository import UserRepository


def test_user_crud() -> None:
    """Test create, read, update, and delete for users."""
    initialize_schema()

    repository = UserRepository()

    user_id = repository.create_user(
        username="AI User",
        email="ai.user@example.com",
    )

    user = repository.get_user(user_id)

    assert user is not None
    assert user["username"] == "AI User"
    assert user["email"] == "ai.user@example.com"

    repository.update_user(
        user_id=user_id,
        username="Updated AI User",
        email="updated@example.com",
    )

    updated_user = repository.get_user(user_id)

    assert updated_user is not None
    assert updated_user["username"] == "Updated AI User"
    assert updated_user["email"] == "updated@example.com"

    repository.delete_user(user_id)

    deleted_user = repository.get_user(user_id)

    assert deleted_user is None
