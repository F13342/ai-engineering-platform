from ai_platform.database.repositories.project_repository import ProjectRepository
from ai_platform.database.repositories.user_repository import UserRepository
from ai_platform.database.schema import initialize_schema


def test_project_crud() -> None:
    """Test project CRUD operations and user relationship."""
    initialize_schema()

    user_repository = UserRepository()
    project_repository = ProjectRepository()

    user_id = user_repository.create_user(
        username="Project User",
        email="project.user@example.com",
    )

    project_id = project_repository.create_project(
        user_id=user_id,
        name="AI Platform Core",
        description="Core project for the AI platform.",
    )

    project = project_repository.get_project(project_id)

    assert project is not None
    assert project["user_id"] == user_id
    assert project["name"] == "AI Platform Core"

    projects = project_repository.get_projects_by_user(user_id)

    assert len(projects) == 1
    assert projects[0]["project_id"] == project_id

    project_repository.update_project(
        project_id=project_id,
        name="AI Platform Core v2",
        description="Updated AI platform project.",
    )

    updated_project = project_repository.get_project(project_id)

    assert updated_project is not None
    assert updated_project["name"] == "AI Platform Core v2"
    assert updated_project["description"] == "Updated AI platform project."

    project_repository.delete_project(project_id)

    deleted_project = project_repository.get_project(project_id)

    assert deleted_project is None

    user_repository.delete_user(user_id)
