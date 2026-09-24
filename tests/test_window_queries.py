from ai_platform.database.queries.window import get_user_activity_ranking
from ai_platform.database.repositories.project_repository import ProjectRepository
from ai_platform.database.repositories.user_repository import UserRepository
from ai_platform.database.schema import initialize_schema


def test_window_function_ranking() -> None:
    """Test ROW_NUMBER and RANK window functions."""
    initialize_schema()

    user_repository = UserRepository()
    project_repository = ProjectRepository()

    user_a = user_repository.create_user(
        username="Ranking User A",
        email="ranking.a@example.com",
    )

    user_b = user_repository.create_user(
        username="Ranking User B",
        email="ranking.b@example.com",
    )

    user_c = user_repository.create_user(
        username="Ranking User C",
        email="ranking.c@example.com",
    )

    for number in range(3):
        project_repository.create_project(
            user_id=user_a,
            name=f"A Project {number}",
            description="Ranking test project.",
        )

    for number in range(2):
        project_repository.create_project(
            user_id=user_b,
            name=f"B Project {number}",
            description="Ranking test project.",
        )

    for number in range(2):
        project_repository.create_project(
            user_id=user_c,
            name=f"C Project {number}",
            description="Ranking test project.",
        )

    results = get_user_activity_ranking()

    result_a = next(
        item for item in results if item["user_id"] == user_a
    )

    result_b = next(
        item for item in results if item["user_id"] == user_b
    )

    result_c = next(
        item for item in results if item["user_id"] == user_c
    )

    assert result_a["project_count"] == 3
    assert result_a["activity_rank"] == 1
    assert result_a["row_number"] == 1

    assert result_b["project_count"] == 2
    assert result_b["activity_rank"] == 2

    assert result_c["project_count"] == 2
    assert result_c["activity_rank"] == 2

    assert {
        result_b["row_number"],
        result_c["row_number"],
    } == {2, 3}

    for user_id in [user_a, user_b, user_c]:
        projects = project_repository.get_projects_by_user(user_id)

        for project in projects:
            project_repository.delete_project(project["project_id"])

        user_repository.delete_user(user_id)
