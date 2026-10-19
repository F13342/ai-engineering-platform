import pytest

pytest.importorskip(
    "torch",
    reason="Install optional ML dependencies or run these tests in Google Colab.",
)

import torch

from ai_platform.machine_learning.language_model.training import (
    LanguageModelTrainingConfig,
    generate_text,
    load_language_model,
    train_language_model,
)


def test_training_config_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match="divisible"):
        LanguageModelTrainingConfig(n_embd=30, n_heads=4)


def test_training_config_rejects_invalid_training_steps():
    with pytest.raises(ValueError, match="positive"):
        LanguageModelTrainingConfig(max_steps=0)


def test_train_save_reload_and_generate(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        ("The quick brown fox jumps over the lazy dog.\\n" * 40),
        encoding="utf-8",
    )
    output_dir = tmp_path / "artifacts"
    config = LanguageModelTrainingConfig(
        block_size=8,
        n_layers=1,
        n_heads=2,
        n_embd=16,
        dropout=0.0,
        batch_size=2,
        max_steps=2,
        learning_rate=1e-3,
        eval_interval=1,
        eval_batches=1,
        train_fraction=0.8,
        seed=17,
        device="cpu",
    )

    result = train_language_model(corpus, output_dir, config)

    assert result["steps"] == 2
    assert (output_dir / "best.pt").is_file()
    assert (output_dir / "last.pt").is_file()
    assert (output_dir / "tokenizer.json").is_file()
    assert (output_dir / "metrics.json").is_file()

    model, tokenizer, checkpoint = load_language_model(
        output_dir / "best.pt", device="cpu"
    )
    assert model.training is False
    assert tokenizer.vocab_size == model.config.vocab_size
    assert checkpoint["step"] in (1, 2)

    generated = generate_text(
        output_dir / "best.pt", "The quick", max_new_tokens=4, device="cpu"
    )
    assert generated.startswith("The quick")
    assert len("The quick") <= len(generated) <= len("The quick") + 4


def test_training_rejects_missing_corpus(tmp_path):
    config = LanguageModelTrainingConfig(
        block_size=8, n_layers=1, n_heads=2, n_embd=16, device="cpu"
    )
    with pytest.raises(FileNotFoundError, match="Corpus file not found"):
        train_language_model(tmp_path / "missing.txt", tmp_path / "artifacts", config)


def test_bpe_training_saves_reloadable_checkpoint_and_generates_text(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        ("Alice saw the rabbit. Alice followed the rabbit.\n" * 80),
        encoding="utf-8",
    )
    output_dir = tmp_path / "bpe_artifacts"
    config = LanguageModelTrainingConfig(
        block_size=8,
        n_layers=1,
        n_heads=2,
        n_embd=16,
        dropout=0.0,
        batch_size=2,
        max_steps=2,
        learning_rate=1e-3,
        eval_interval=1,
        eval_batches=1,
        train_fraction=0.8,
        seed=9,
        tokenizer_type="bpe",
        tokenizer_vocab_size=48,
        tokenizer_min_frequency=2,
        device="cpu",
    )

    result = train_language_model(corpus, output_dir, config)

    assert result["steps"] == 2
    assert (output_dir / "best.pt").is_file()
    assert (output_dir / "tokenizer.json").is_file()

    tokenizer_json = __import__("json").loads(
        (output_dir / "tokenizer.json").read_text(encoding="utf-8")
    )
    assert tokenizer_json["tokenizer_type"] == "bpe_char"

    model, tokenizer, checkpoint = load_language_model(
        output_dir / "best.pt", device="cpu"
    )
    assert model.training is False
    assert tokenizer.vocab_size == model.config.vocab_size
    assert checkpoint["step"] in (1, 2)

    generated = generate_text(
        output_dir / "best.pt",
        "Alice",
        max_new_tokens=3,
        device="cpu",
    )
    assert generated.startswith("Alice")


def test_resume_continues_step_count_and_optimizer_state(tmp_path):
    from dataclasses import replace

    corpus = tmp_path / "resume_corpus.txt"
    corpus.write_text(
        ("Alice sees the rabbit and follows it into the garden.\n" * 100),
        encoding="utf-8",
    )
    output_dir = tmp_path / "resumable_artifacts"

    initial_config = LanguageModelTrainingConfig(
        block_size=8,
        n_layers=1,
        n_heads=2,
        n_embd=16,
        dropout=0.0,
        batch_size=2,
        max_steps=2,
        learning_rate=1e-3,
        eval_interval=1,
        eval_batches=1,
        train_fraction=0.8,
        seed=43,
        tokenizer_type="bpe",
        tokenizer_vocab_size=48,
        tokenizer_min_frequency=2,
        device="cpu",
    )

    first_result = train_language_model(corpus, output_dir, initial_config)
    assert first_result["steps"] == 2

    last_checkpoint = output_dir / "last.pt"
    checkpoint = torch.load(
        last_checkpoint, map_location="cpu", weights_only=True
    )
    assert checkpoint["step"] == 2
    assert "optimizer_state_dict" in checkpoint
    assert "cpu_rng_state" in checkpoint
    assert "corpus_sha256" in checkpoint

    continued_config = replace(initial_config, max_steps=4)
    resumed_result = train_language_model(
        corpus,
        output_dir,
        continued_config,
        resume_from=last_checkpoint,
    )

    assert resumed_result["resumed_from_step"] == 2
    assert resumed_result["steps"] == 4

    final_checkpoint = torch.load(
        last_checkpoint, map_location="cpu", weights_only=True
    )
    assert final_checkpoint["step"] == 4

    metrics = json.loads(
        (output_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert [item["step"] for item in metrics] == [1, 2, 3, 4]


def test_resume_rejects_changed_corpus(tmp_path):
    from dataclasses import replace

    corpus = tmp_path / "resume_corpus.txt"
    corpus.write_text(
        ("The quick brown fox jumps over the lazy dog.\n" * 100),
        encoding="utf-8",
    )
    output_dir = tmp_path / "resumable_artifacts"

    config = LanguageModelTrainingConfig(
        block_size=8,
        n_layers=1,
        n_heads=2,
        n_embd=16,
        dropout=0.0,
        batch_size=2,
        max_steps=1,
        learning_rate=1e-3,
        eval_interval=1,
        eval_batches=1,
        train_fraction=0.8,
        seed=47,
        tokenizer_type="character",
        device="cpu",
    )
    train_language_model(corpus, output_dir, config)
    last_checkpoint = output_dir / "last.pt"

    corpus.write_text(
        corpus.read_text(encoding="utf-8") + "New different corpus material.\n",
        encoding="utf-8",
    )

    continued_config = replace(config, max_steps=2)
    with pytest.raises(ValueError, match="different corpus"):
        train_language_model(
            corpus,
            output_dir,
            continued_config,
            resume_from=last_checkpoint,
        )
