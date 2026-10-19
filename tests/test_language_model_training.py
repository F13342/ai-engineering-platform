import pytest

pytest.importorskip(
    "torch",
    reason="Install optional ML dependencies or run these tests in Google Colab.",
)

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
