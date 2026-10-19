"""Reproducible held-out evaluation for language-model checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .training import load_language_model


@torch.no_grad()
def evaluate_language_model_checkpoint(
    checkpoint_path: str | Path,
    validation_text: str,
    device: str = "auto",
) -> dict[str, Any]:
    """Evaluate next-token NLL and bits per character on held-out text.

    Consecutive non-overlapping input blocks are used. Every token after the
    first encoded token is scored exactly once; context is reset at block
    boundaries, making results reproducible with the same block size.
    """
    if not isinstance(validation_text, str):
        raise TypeError("validation_text must be a string.")
    if not validation_text:
        raise ValueError("validation_text cannot be empty.")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, tokenizer, checkpoint = load_language_model(
        checkpoint_path,
        device=device,
    )
    model.eval()
    target_device = next(model.parameters()).device

    # The tokenizer normalizes BPE text to Unicode NFC before encoding.
    tokenizer_metadata_path = checkpoint_path.parent / "tokenizer.json"
    tokenizer_metadata = json.loads(
        tokenizer_metadata_path.read_text(encoding="utf-8")
    )
    if tokenizer_metadata.get("tokenizer_type") == "bpe_char":
        validation_text = tokenizer._normalize(validation_text)

    token_ids = tokenizer.encode(validation_text)
    if len(token_ids) < 2:
        raise ValueError("Validation text must encode to at least two tokens.")

    ids = torch.tensor(token_ids, dtype=torch.long, device=target_device)
    block_size = int(checkpoint["model_config"]["block_size"])
    total_nll = 0.0
    predicted_tokens = 0

    for start in range(0, ids.numel() - 1, block_size):
        end = min(start + block_size, ids.numel() - 1)
        inputs = ids[start:end].unsqueeze(0)
        targets = ids[start + 1:end + 1].unsqueeze(0)

        logits, _ = model(inputs)
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="sum",
        )

        total_nll += float(loss_sum.item())
        predicted_tokens += targets.numel()

    if predicted_tokens != len(token_ids) - 1:
        raise RuntimeError("Evaluation did not score every next-token target once.")

    average_nll = total_nll / predicted_tokens
    bits_per_character = total_nll / math.log(2.0) / len(validation_text)

    tokenizer_kind = tokenizer_metadata.get("tokenizer_type", "unknown")
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "tokenizer_type": tokenizer_kind,
        "vocabulary_size": int(tokenizer.vocab_size),
        "validation_characters": len(validation_text),
        "validation_tokens": len(token_ids),
        "predicted_tokens": predicted_tokens,
        "tokens_per_character": len(token_ids) / len(validation_text),
        "average_nll_per_token_nats": average_nll,
        "bits_per_character": bits_per_character,
        "evaluation_method": "non-overlapping next-token blocks",
        "block_size": block_size,
    }
