import json

from ai_platform.projects.project import Project
from ai_platform.projects.repository import ProjectRepository


def test_repository_saves_and_loads_projects(tmp_path) -> None:
    file_path = tmp_path / "projects.json"
    repository = ProjectRepository(file_path)

    project = Project(
        name="Test Project",
        description="Test description",
    )

    repository.save_projects([project])

    loaded_projects = repository.load_projects()

    assert len(loaded_projects) == 1
    assert loaded_projects[0].name == "Test Project"
    assert loaded_projects[0].description == "Test description"


def test_repository_creates_valid_json_file(tmp_path) -> None:
    file_path = tmp_path / "projects.json"
    repository = ProjectRepository(file_path)

    repository.save_projects(
        [
            Project(
                name="Project One",
                description="First project",
            )
        ]
    )

    data = json.loads(file_path.read_text())

    assert data == [
        {
            "name": "Project One",
            "description": "First project",
        }
    ]
def test_repository_returns_empty_list_when_file_does_not_exist(tmp_path) -> None:
    file_path = tmp_path / "projects.json"
    repository = ProjectRepository(file_path)

    projects = repository.load_projects()

    assert projects == []
