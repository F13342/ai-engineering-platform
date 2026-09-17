from .project import Project


class ProjectRegistry:
    """Manage projects in the AI Engineering Platform."""

    def __init__(self) -> None:
        self._projects: dict[str, Project] = {}

    def add_project(self, project: Project) -> None:
        """Add a project to the registry."""
        self._projects[project.name] = project

    def get_project(self, name: str) -> Project | None:
        """Return a project by name, or None if it does not exist."""
        return self._projects.get(name)

    def list_projects(self) -> list[Project]:
        """Return all registered projects."""
        return list(self._projects.values())
