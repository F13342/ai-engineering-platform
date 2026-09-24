from pathlib import Path

from ai_platform.database.analytics.export import (
    export_user_analytical_dataset,
)
from ai_platform.database.repositories.project_repository import ProjectRepository
from ai_platform.database.repositories.user_repository import UserRepository
from ai_platform.database.schema import initialize_schema


def test_analytical_dataset_real_export(tmp_path: Path) -> None:
    """Test building and exporting the real analytical dataset."""
    initialize_schema()

    user_repository = UserRepository()
    project_repository = ProjectRepository()

    user_id = user_repository.create_user(
        username="Export User",
        email="export.user@example.com",
    )

    project_repository.create_project(
        user_id=user_id,
        name="Export Project",
        description="Analytical export test.",
    )

    output_file = tmp_path / "user_activity_analytics.csv"

    export_user_analytical_dataset(output_file)

    assert output_file.exists()

    content = output_file.read_text(encoding="utf-8")

    assert "user_id,username,project_count" in content
    assert "Export User" in content

    projects = project_repository.get_projects_by_user(user_id)

    for project in projects:
        project_repository.delete_project(project["project_id"])

    user_repository.delete_user(user_id)
