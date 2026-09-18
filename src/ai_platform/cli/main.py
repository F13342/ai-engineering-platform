import argparse


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

    return parser


def main() -> None:
    """Run the command-line interface."""
    parser = create_parser()
    parser.parse_args()


if __name__ == "__main__":
    main()
