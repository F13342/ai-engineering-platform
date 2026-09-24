from pathlib import Path

from ai_platform.data_engineering.sources.sql_analytics_source import (
    load_sql_analytical_data,
    save_sql_analytical_data,
)
from ai_platform.database.repositories.project_repository import ProjectRepository
from ai_platform.database.repositories.user_repository import UserRepository
from ai_platform.database.schema import initialize_schema


def test_sql_analytics_source(tmp_path: Path) -> None:
    """Test loading and saving SQL analytical data."""
    initialize_schema()

    user_repository = UserRepository()
    project_repository = ProjectRepository()

    user_id = user_repository.create_user(
        username="DE Source User",
        email="de.source@example.com",
    )

    project_repository.create_project(
        user_id=user_id,
        name="Data Pipeline Project",
        description="Project used to test SQL analytical ingestion.",
    )

    data = load_sql_analytical_data()

    assert not data.empty

    expected_columns = {
        "user_id",
        "username",
        "project_count",
        "conversation_count",
        "message_count",
        "document_count",
        "activity_total",
        "activity_rank",
        "average_activity",
        "difference_from_average",
        "activity_level",
    }

    assert expected_columns.issubset(data.columns)

    result = data.loc[data["user_id"] == user_id].iloc[0]

    assert result["project_count"] == 1
    assert result["activity_total"] == 1

    output_file = tmp_path / "sql_analytical_dataset.csv"

    save_sql_analytical_data(output_file)

    assert output_file.exists()

    projects = project_repository.get_projects_by_user(user_id)

    for project in projects:
        project_repository.delete_project(project["project_id"])

    user_repository.delete_user(user_id)
