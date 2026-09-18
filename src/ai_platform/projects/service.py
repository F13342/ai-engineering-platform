from .project import Project
from .registry import ProjectRegistry


class ProjectService:
    """Provide project management operations."""

    def __init__(self, registry: ProjectRegistry) -> None:
        self._registry = registry

    def add_project(self, project: Project) -> None:
        """Add a project to the registry."""
        self._registry.add_project(project)

    def list_projects(self) -> list[Project]:
        """Return all registered projects."""
        return self._registry.list_projects()
