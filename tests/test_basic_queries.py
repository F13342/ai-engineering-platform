from ai_platform.database.queries.basic import (
    get_projects_with_users,
    get_user_projects,
    search_projects,
)
from ai_platform.database.repositories.project_repository import ProjectRepository
from ai_platform.database.repositories.user_repository import UserRepository
from ai_platform.database.schema import initialize_schema


def test_basic_sql_queries() -> None:
    """Test SELECT, WHERE, ORDER BY, and INNER JOIN queries."""
    initialize_schema()

    user_repository = UserRepository()
    project_repository = ProjectRepository()

    user_id = user_repository.create_user(
        username="SQL User",
        email="sql.user@example.com",
    )

    project_id = project_repository.create_project(
        user_id=user_id,
        name="RAG Engine",
        description="Retrieval augmented generation module.",
    )

    projects = get_user_projects(user_id)

    assert len(projects) == 1
    assert projects[0]["project_id"] == project_id
    assert projects[0]["name"] == "RAG Engine"

    search_results = search_projects("RAG")

    assert any(
        project["project_id"] == project_id
        for project in search_results
    )

    joined_projects = get_projects_with_users()

    matching_project = next(
        project
        for project in joined_projects
        if project["project_id"] == project_id
    )

    assert matching_project["username"] == "SQL User"
    assert matching_project["email"] == "sql.user@example.com"

    project_repository.delete_project(project_id)
    user_repository.delete_user(user_id)
