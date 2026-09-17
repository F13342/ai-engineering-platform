from ai_platform.projects.project import Project
from ai_platform.projects.registry import ProjectRegistry


def test_project_can_be_added_and_retrieved() -> None:
    registry = ProjectRegistry()
    project = Project(
        name="AI Engineering Platform",
        description="Learning and development platform",
    )

    registry.add_project(project)

    assert registry.get_project("AI Engineering Platform") is project


def test_missing_project_returns_none() -> None:
    registry = ProjectRegistry()

    assert registry.get_project("Unknown Project") is None


def test_registry_lists_all_projects() -> None:
    registry = ProjectRegistry()
    first_project = Project("Project One", "First project")
    second_project = Project("Project Two", "Second project")

    registry.add_project(first_project)
    registry.add_project(second_project)

    assert registry.list_projects() == [first_project, second_project]
