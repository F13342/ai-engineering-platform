from .project import Project


def format_project_list(projects: list[Project]) -> str:
    """Format projects for display in the command-line interface."""
    if not projects:
        return "No projects found."

    lines = ["Projects:"]

    for project in projects:
        lines.append(f"- {project.name}: {project.description}")

    return "\n".join(lines)
