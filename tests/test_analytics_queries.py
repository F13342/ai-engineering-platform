from ai_platform.database.queries.analytics import (
    get_user_project_statistics,
    get_users_with_minimum_projects,
)
from ai_platform.database.repositories.project_repository import ProjectRepository
from ai_platform.database.repositories.user_repository import UserRepository
from ai_platform.database.schema import initialize_schema


def test_group_by_and_having_queries() -> None:
    """Test GROUP BY, COUNT, and HAVING."""
    initialize_schema()

    user_repository = UserRepository()
    project_repository = ProjectRepository()

    user_id = user_repository.create_user(
        username="Analytics User",
        email="analytics.user@example.com",
    )

    project_repository.create_project(
        user_id=user_id,
        name="Project One",
        description="First project.",
    )

    project_repository.create_project(
        user_id=user_id,
        name="Project Two",
        description="Second project.",
    )

    statistics = get_user_project_statistics()

    user_statistics = next(
        item
        for item in statistics
        if item["user_id"] == user_id
    )

    assert user_statistics["username"] == "Analytics User"
    assert user_statistics["project_count"] == 2

    filtered_statistics = get_users_with_minimum_projects(2)

    matching_user = next(
        item
        for item in filtered_statistics
        if item["user_id"] == user_id
    )

    assert matching_user["project_count"] == 2

    project_ids = project_repository.get_projects_by_user(user_id)

    for project in project_ids:
        project_repository.delete_project(project["project_id"])

    user_repository.delete_user(user_id)
