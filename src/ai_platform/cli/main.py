import argparse

from ai_platform.projects.formatter import format_project_list
from ai_platform.projects.project import Project
from ai_platform.projects.registry import ProjectRegistry
from ai_platform.projects.service import ProjectService


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="ai-platform",
        description="AI Engineering Platform command-line interface",
    )

    subparsers = parser.add_subparsers(dest="command")

    project_parser = subparsers.add_parser(
        "project",
        help="Manage projects",
    )

    project_subparsers = project_parser.add_subparsers(
        dest="project_command"
    )

    project_subparsers.add_parser(
        "list",
        help="List all projects",
    )

    add_parser = project_subparsers.add_parser(
        "add",
        help="Add a new project",
    )

    add_parser.add_argument(
        "--name",
        help="Project name",
    )

    add_parser.add_argument(
        "--description",
        help="Project description",
    )

    return parser


def create_project_service() -> ProjectService:
    """Create the project service with initial project data."""
    registry = ProjectRegistry()

    registry.add_project(
        Project(
            name="AI Engineering Platform",
            description="Learning and development platform",
        )
    )

    return ProjectService(registry)


def main() -> None:
    """Run the command-line interface."""
    parser = create_parser()
    arguments = parser.parse_args()

    if arguments.command == "project":
        service = create_project_service()

        if arguments.project_command == "list":
            projects = service.list_projects()
            print(format_project_list(projects))

        elif arguments.project_command == "add":
            project = Project(
                name=arguments.name,
                description=arguments.description or "",
            )

            service.add_project(project)

            print(f"Project added: {project.name}")


if __name__ == "__main__":
    main()
