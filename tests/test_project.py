import pytest

from ai_platform.projects.project import Project


def test_project_can_be_created() -> None:
    project = Project(
        name="AI Engineering Platform",
        description="Learning and development platform",
    )

    assert project.name == "AI Engineering Platform"
    assert project.description == "Learning and development platform"


def test_project_requires_name() -> None:
    with pytest.raises(ValueError, match="Project name cannot be empty"):
        Project(
            name="",
            description="Project without a name",
        )


def test_project_requires_description() -> None:
    with pytest.raises(
        ValueError,
        match="Project description cannot be empty",
    ):
        Project(
            name="AI Engineering Platform",
            description="",
        )
