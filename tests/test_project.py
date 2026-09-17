from ai_platform.projects.project import Project


def test_project_can_be_created() -> None:
    project = Project(
        name="AI Engineering Platform",
        description="Learning and development platform",
    )

    assert project.name == "AI Engineering Platform"
    assert project.description == "Learning and development platform"
