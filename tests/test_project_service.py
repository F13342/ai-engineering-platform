from ai_platform.projects.project import Project
from ai_platform.projects.registry import ProjectRegistry
from ai_platform.projects.service import ProjectService


def test_service_lists_registered_projects() -> None:
    registry = ProjectRegistry()
    service = ProjectService(registry)
    project = Project("Project One", "First project")

    registry.add_project(project)

    assert service.list_projects() == [project]
