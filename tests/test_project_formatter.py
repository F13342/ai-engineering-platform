from ai_platform.projects.formatter import format_project_list
from ai_platform.projects.project import Project


def test_format_empty_project_list() -> None:
    assert format_project_list([]) == "No projects found."


def test_format_project_list() -> None:
    projects = [
        Project("Project One", "First project"),
        Project("Project Two", "Second project"),
    ]

    result = format_project_list(projects)

    assert result == (
        "Projects:\n"
        "- Project One: First project\n"
        "- Project Two: Second project"
    )
