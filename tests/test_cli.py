from ai_platform.cli.main import create_parser


def test_cli_parser_has_expected_program_name() -> None:
    parser = create_parser()

    assert parser.prog == "ai-platform"


def test_cli_supports_project_list_command() -> None:
    parser = create_parser()

    arguments = parser.parse_args(["project", "list"])

    assert arguments.command == "project"
    assert arguments.project_command == "list"
def test_cli_supports_project_add_command() -> None:
    parser = create_parser()

    arguments = parser.parse_args(["project", "add"])

    assert arguments.command == "project"
    assert arguments.project_command == "add"
def test_cli_add_command_requires_project_name() -> None:
    parser = create_parser()

    arguments = parser.parse_args(
        ["project", "add", "--name", "Test Project"]
    )

    assert arguments.name == "Test Project"
def test_cli_add_command_accepts_project_description() -> None:
    parser = create_parser()

    arguments = parser.parse_args(
        [
            "project",
            "add",
            "--name",
            "Test Project",
            "--description",
            "Test description",
        ]
    )

    assert arguments.name == "Test Project"
    assert arguments.description == "Test description"
def test_cli_add_command_creates_project() -> None:
    parser = create_parser()

    arguments = parser.parse_args(
        [
            "project",
            "add",
            "--name",
            "Test Project",
            "--description",
            "Test description",
        ]
    )

    assert arguments.name == "Test Project"
    assert arguments.description == "Test description"
