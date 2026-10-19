import math

import pytest

pytest.importorskip(
    "torch",
    reason="Install optional ML dependencies or run in Google Colab.",
)

from ai_platform.machine_learning.language_model.training import (
    LanguageModelTrainingConfig,
    train_language_model,
)
from ai_platform.machine_learning.language_model.evaluation import (
    evaluate_language_model_checkpoint,
)


@pytest.mark.parametrize("tokenizer_type", ["character", "bpe"])
def test_evaluation_returns_valid_metrics_for_each_tokenizer(
    tmp_path, tokenizer_type
):
    corpus_text = (
        "Alice sees the rabbit. Alice follows the rabbit!\n" * 80
    )
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(corpus_text, encoding="utf-8")
    output_dir = tmp_path / f"artifacts_{tokenizer_type}"

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
        seed=29,
        tokenizer_type=tokenizer_type,
        tokenizer_vocab_size=64,
        tokenizer_min_frequency=2,
        device="cpu",
    )

    train_language_model(corpus_path, output_dir, config)
    split_at = int(len(corpus_text) * config.train_fraction)
    validation_text = corpus_text[split_at:]

    metrics = evaluate_language_model_checkpoint(
        output_dir / "best.pt",
        validation_text,
        device="cpu",
    )

    assert metrics["tokenizer_type"] == (
        "bpe_char" if tokenizer_type == "bpe" else "character"
    )
    assert metrics["validation_characters"] == len(validation_text)
    assert metrics["predicted_tokens"] == metrics["validation_tokens"] - 1
    assert metrics["vocabulary_size"] > 4
    assert math.isfinite(metrics["average_nll_per_token_nats"])
    assert math.isfinite(metrics["bits_per_character"])
    assert metrics["bits_per_character"] > 0
    assert metrics["block_size"] == config.block_size


def test_evaluation_rejects_empty_validation_text(tmp_path):
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluate_language_model_checkpoint(
            tmp_path / "missing.pt",
            "",
            device="cpu",
        )
