import pytest

pytest.importorskip(
    "torch",
    reason="Install the optional ML dependencies with pip install -e .[ml,dev].",
)

import torch

from ai_platform.machine_learning.language_model.model import (
    TinyTransformerLM,
    TransformerConfig,
)


def small_config(**overrides):
    settings = {
        "vocab_size": 32,
        "block_size": 16,
        "n_layers": 2,
        "n_heads": 4,
        "n_embd": 32,
        "dropout": 0.0,
    }
    settings.update(overrides)
    return TransformerConfig(**settings)


def test_forward_produces_logits_loss_and_gradients():
    torch.manual_seed(7)
    model = TinyTransformerLM(small_config())
    token_ids = torch.tensor(
        [[1, 2, 3, 4], [4, 3, 2, 1]],
        dtype=torch.long,
    )

    logits, loss = model(token_ids, targets=token_ids)

    assert logits.shape == (2, 4, 32)
    assert loss is not None
    assert torch.isfinite(loss)

    loss.backward()
    parameters_with_gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert parameters_with_gradients
    assert all(gradient is not None for gradient in parameters_with_gradients)
    assert all(torch.isfinite(gradient).all() for gradient in parameters_with_gradients)


def test_attention_is_causal():
    torch.manual_seed(11)
    model = TinyTransformerLM(small_config())
    model.eval()

    sequence_a = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    sequence_b = torch.tensor([[1, 2, 9, 8]], dtype=torch.long)

    with torch.no_grad():
        logits_a, _ = model(sequence_a)
        logits_b, _ = model(sequence_b)

    # Changing future tokens must not change logits at earlier positions.
    torch.testing.assert_close(logits_a[:, :2], logits_b[:, :2])


def test_generation_appends_valid_tokens():
    torch.manual_seed(13)
    model = TinyTransformerLM(small_config())
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    generated = model.generate(
        prompt,
        max_new_tokens=5,
        temperature=1.0,
        top_k=8,
    )

    assert generated.shape == (1, 8)
    assert generated.dtype == torch.long
    assert torch.all(generated >= 0)
    assert torch.all(generated < model.config.vocab_size)
    torch.testing.assert_close(generated[:, :3], prompt)


def test_model_rejects_context_longer_than_block_size():
    model = TinyTransformerLM(small_config(block_size=4))
    too_long = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    with pytest.raises(ValueError, match="block_size"):
        model(too_long)


def test_config_rejects_incompatible_attention_dimensions():
    with pytest.raises(ValueError, match="divisible"):
        small_config(n_embd=30, n_heads=4)
