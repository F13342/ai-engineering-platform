import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ai_platform.projects.formatter import format_project_list
from ai_platform.projects.project import Project
from ai_platform.projects.repository import ProjectRepository
from ai_platform.projects.service import ProjectService


PROJECTS_FILE = Path("data/projects.json")


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="ai-platform",
        description="AI Engineering Platform command-line interface",
    )
    subparsers = parser.add_subparsers(dest="command")

    project_parser = subparsers.add_parser(
        "project", help="Manage projects"
    )
    project_subparsers = project_parser.add_subparsers(
        dest="project_command"
    )
    project_subparsers.add_parser("list", help="List all projects")

    add_parser = project_subparsers.add_parser(
        "add", help="Add a new project"
    )
    add_parser.add_argument("--name", help="Project name")
    add_parser.add_argument("--description", help="Project description")

    train = subparsers.add_parser(
        "train", help="Train or resume a language model"
    )
    train.add_argument("--corpus", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--resume-from", type=Path)
    train.add_argument("--max-steps", type=int, default=1000)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--block-size", type=int, default=128)
    train.add_argument("--n-layers", type=int, default=4)
    train.add_argument("--n-heads", type=int, default=4)
    train.add_argument("--n-embd", type=int, default=128)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--eval-interval", type=int, default=100)
    train.add_argument("--eval-batches", type=int, default=20)
    train.add_argument("--train-fraction", type=float, default=0.9)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--tokenizer-type",
        choices=("character", "bpe"),
        default="character",
    )
    train.add_argument("--tokenizer-vocab-size", type=int, default=512)
    train.add_argument("--tokenizer-min-frequency", type=int, default=2)
    train.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate a saved language model"
    )
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--text-file", type=Path, required=True)
    evaluate.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )

    generate = subparsers.add_parser(
        "generate", help="Generate text from a saved language model"
    )
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=100)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-k", type=int)
    generate.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )

    return parser


def create_project_service() -> ProjectService:
    """Create the project service with JSON persistence."""
    return ProjectService(ProjectRepository(PROJECTS_FILE))


def main(argv: Sequence[str] | None = None) -> int:
    """Run project management or language-model commands."""
    arguments = create_parser().parse_args(argv)

    if arguments.command == "project":
        service = create_project_service()

        if arguments.project_command == "list":
            print(format_project_list(service.list_projects()))
        elif arguments.project_command == "add":
            project = Project(
                name=arguments.name,
                description=arguments.description or "",
            )
            service.add_project(project)
            print(f"Project added: {project.name}")
        return 0

    if arguments.command == "train":
        from ai_platform.machine_learning.language_model.training import (
            LanguageModelTrainingConfig,
            train_language_model,
        )

        config = LanguageModelTrainingConfig(
            block_size=arguments.block_size,
            n_layers=arguments.n_layers,
            n_heads=arguments.n_heads,
            n_embd=arguments.n_embd,
            dropout=arguments.dropout,
            batch_size=arguments.batch_size,
            max_steps=arguments.max_steps,
            learning_rate=arguments.learning_rate,
            weight_decay=arguments.weight_decay,
            eval_interval=arguments.eval_interval,
            eval_batches=arguments.eval_batches,
            train_fraction=arguments.train_fraction,
            grad_clip=arguments.grad_clip,
            seed=arguments.seed,
            tokenizer_type=arguments.tokenizer_type,
            tokenizer_vocab_size=arguments.tokenizer_vocab_size,
            tokenizer_min_frequency=arguments.tokenizer_min_frequency,
            device=arguments.device,
        )
        result = train_language_model(
            arguments.corpus,
            arguments.output_dir,
            config,
            resume_from=arguments.resume_from,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if arguments.command == "evaluate":
        from ai_platform.machine_learning.language_model.evaluation import (
            evaluate_language_model_checkpoint,
        )

        validation_text = arguments.text_file.read_text(encoding="utf-8")
        result = evaluate_language_model_checkpoint(
            arguments.checkpoint,
            validation_text,
            device=arguments.device,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if arguments.command == "generate":
        from ai_platform.machine_learning.language_model.training import (
            generate_text,
        )

        result = generate_text(
            checkpoint_path=arguments.checkpoint,
            prompt=arguments.prompt,
            max_new_tokens=arguments.max_new_tokens,
            temperature=arguments.temperature,
            top_k=arguments.top_k,
            device=arguments.device,
        )
        print(result)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
