from ai_platform.cli.main import create_parser


def test_cli_parser_has_expected_program_name() -> None:
    parser = create_parser()

    assert parser.prog == "ai-platform"
