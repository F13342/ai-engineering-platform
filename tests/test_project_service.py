from pathlib import Path

from ai_platform.projects.project import Project
from ai_platform.projects.repository import ProjectRepository
from ai_platform.projects.service import ProjectService


def test_service_lists_saved_projects(tmp_path: Path) -> None:
    file_path = tmp_path / "projects.json"
    repository = ProjectRepository(file_path)
    service = ProjectService(repository)

    project = Project(
        name="Project One",
        description="First project",
    )

    repository.save_projects([project])

    assert service.list_projects() == [project]


def test_service_adds_project(tmp_path: Path) -> None:
    file_path = tmp_path / "projects.json"
    repository = ProjectRepository(file_path)
    service = ProjectService(repository)

    project = Project(
        name="Project Two",
        description="Second project",
    )

    service.add_project(project)

    assert service.list_projects() == [project]


def test_service_adds_project_with_name_and_description(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "projects.json"
    repository = ProjectRepository(file_path)
    service = ProjectService(repository)

    service.add_project(
        Project(
            name="Test Project",
            description="Test description",
        )
    )

    project = service.list_projects()[0]

    assert project.name == "Test Project"
    assert project.description == "Test description"


def test_service_saves_project_through_repository(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "projects.json"
    repository = ProjectRepository(file_path)
    service = ProjectService(repository)

    project = Project(
        name="Saved Project",
        description="Persistent project",
    )

    service.add_project(project)

    loaded_projects = repository.load_projects()

    assert loaded_projects[0].name == "Saved Project"
    assert loaded_projects[0].description == "Persistent project"
