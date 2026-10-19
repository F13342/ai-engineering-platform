from pathlib import Path

import pytest

from ai_platform.cli.main import create_parser


def test_train_command_parses_configuration_and_resume() -> None:
    args = create_parser().parse_args([
        "train",
        "--corpus", "data/train.txt",
        "--output-dir", "runs/experiment",
        "--max-steps", "50",
        "--tokenizer-type", "bpe",
        "--resume-from", "runs/experiment/last.pt",
    ])

    assert args.command == "train"
    assert args.corpus == Path("data/train.txt")
    assert args.output_dir == Path("runs/experiment")
    assert args.max_steps == 50
    assert args.tokenizer_type == "bpe"
    assert args.resume_from == Path("runs/experiment/last.pt")


def test_evaluate_command_parses_checkpoint_and_text_file() -> None:
    args = create_parser().parse_args([
        "evaluate",
        "--checkpoint", "runs/experiment/best.pt",
        "--text-file", "data/validation.txt",
        "--device", "cpu",
    ])

    assert args.command == "evaluate"
    assert args.checkpoint == Path("runs/experiment/best.pt")
    assert args.text_file == Path("data/validation.txt")
    assert args.device == "cpu"


def test_generate_command_parses_generation_options() -> None:
    args = create_parser().parse_args([
        "generate",
        "--checkpoint", "runs/experiment/best.pt",
        "--prompt", "Once upon a time",
        "--max-new-tokens", "40",
        "--temperature", "0.8",
        "--top-k", "20",
    ])

    assert args.command == "generate"
    assert args.prompt == "Once upon a time"
    assert args.max_new_tokens == 40
    assert args.temperature == pytest.approx(0.8)
    assert args.top_k == 20


def test_train_rejects_unknown_tokenizer() -> None:
    with pytest.raises(SystemExit):
        create_parser().parse_args([
            "train",
            "--corpus", "train.txt",
            "--output-dir", "runs/test",
            "--tokenizer-type", "unknown",
        ])


def test_existing_project_list_command_is_preserved() -> None:
    args = create_parser().parse_args(["project", "list"])
    assert args.command == "project"
    assert args.project_command == "list"
