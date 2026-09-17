import argparse


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    return argparse.ArgumentParser(
        prog="ai-platform",
        description="AI Engineering Platform command-line interface",
    )


def main() -> None:
    """Run the command-line interface."""
    parser = create_parser()
    parser.parse_args()


if __name__ == "__main__":
    main()
