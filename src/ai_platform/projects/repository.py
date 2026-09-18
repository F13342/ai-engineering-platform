import json
from pathlib import Path

from .project import Project


class ProjectRepository:
    """Persist projects using a JSON file."""

    def __init__(self, file_path: Path) -> None:
        self._file_path = file_path

    def save_projects(self, projects: list[Project]) -> None:
        """Save projects to the JSON file."""
        data = [
            {
                "name": project.name,
                "description": project.description,
            }
            for project in projects
        ]

        self._file_path.write_text(
            json.dumps(data, indent=4),
            encoding="utf-8",
        )

    def load_projects(self) -> list[Project]:
        """Load projects from the JSON file."""
        if not self._file_path.exists():
            return []

        data = json.loads(
            self._file_path.read_text(encoding="utf-8")
        )

        return [
            Project(
                name=item["name"],
                description=item["description"],
            )
            for item in data
        ]
