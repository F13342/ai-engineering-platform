from .project import Project
from .repository import ProjectRepository


class ProjectService:
    """Provide project management operations."""

    def __init__(self, repository: ProjectRepository) -> None:
        self._repository = repository

    def add_project(self, project: Project) -> None:
        """Add a project and persist it."""
        projects = self._repository.load_projects()
        projects.append(project)
        self._repository.save_projects(projects)

    def list_projects(self) -> list[Project]:
        """Return all saved projects."""
        return self._repository.load_projects()
