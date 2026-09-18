class Project:
    """Represent a project managed by the AI Engineering Platform."""

    def __init__(self, name: str, description: str) -> None:
        if not name.strip():
            raise ValueError("Project name cannot be empty")

        if not description.strip():
            raise ValueError("Project description cannot be empty")

        self.name = name
        self.description = description

    def __eq__(self, other: object) -> bool:
        """Compare projects by name and description."""
        if not isinstance(other, Project):
            return NotImplemented

        return (
            self.name == other.name
            and self.description == other.description
        )
