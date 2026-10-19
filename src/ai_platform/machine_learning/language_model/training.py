"""Training and checkpoint utilities for the from-scratch language model."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .model import TinyTransformerLM, TransformerConfig
from .tokenizer import CharacterTokenizer


@dataclass(frozen=True)
class LanguageModelTrainingConfig:
    """Configuration for a reproducible language-model training run."""

    block_size: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    batch_size: int = 16
    max_steps: int = 1000
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    eval_interval: int = 100
    eval_batches: int = 10
    train_fraction: float = 0.9
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2.")
        if self.n_layers < 1 or self.n_heads < 1 or self.n_embd < 1:
            raise ValueError("Model dimensions must be positive.")
        if self.n_embd % self.n_heads:
            raise ValueError("n_embd must be divisible by n_heads.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.batch_size < 1 or self.max_steps < 1:
            raise ValueError("batch_size and max_steps must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("weight_decay must be non-negative and grad_clip positive.")
        if self.eval_interval < 1 or self.eval_batches < 1:
            raise ValueError("Evaluation settings must be positive.")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1.")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps.")


def resolve_device(requested: str = "auto") -> torch.device:
    """Choose an available compute device without requiring a GPU."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but no CUDA GPU is available.")

    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise ValueError("MPS was requested, but it is not available.")

    return torch.device(requested)


def _make_batch(data: Tensor, batch_size: int, block_size: int,
                device: torch.device) -> tuple[Tensor, Tensor]:
    if data.numel() <= block_size:
        raise ValueError("Each corpus split must contain more tokens than block_size.")

    starts = torch.randint(0, data.numel() - block_size, (batch_size,))
    x = torch.stack([data[i.item():i.item() + block_size] for i in starts])
    y = torch.stack([data[i.item() + 1:i.item() + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def _estimate_loss(
    model: TinyTransformerLM,
    data: Tensor,
    config: LanguageModelTrainingConfig,
    device: torch.device,
) -> float:
    model.eval()
    values: list[float] = []

    for _ in range(config.eval_batches):
        x, y = _make_batch(data, config.batch_size, config.block_size, device)
        _, loss = model(x, targets=y)
        if loss is None:
            raise RuntimeError("The model did not return a training loss.")
        values.append(float(loss.item()))

    model.train()
    return sum(values) / len(values)


def train_language_model(
    corpus_path: str | Path,
    output_dir: str | Path,
    config: LanguageModelTrainingConfig | None = None,
) -> dict[str, Any]:
    """Train from random initialization and save model/tokenizer checkpoints."""
    config = config or LanguageModelTrainingConfig()
    corpus_path = Path(corpus_path)
    output_dir = Path(output_dir)

    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    text = corpus_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("The training corpus cannot be empty or whitespace-only.")

    tokenizer = CharacterTokenizer.fit(text)
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_at = int(token_ids.numel() * config.train_fraction)
    train_data = token_ids[:split_at]
    val_data = token_ids[split_at:]

    if train_data.numel() <= config.block_size or val_data.numel() <= config.block_size:
        minimum = math.ceil((config.block_size + 1) / config.train_fraction)
        raise ValueError(
            "Corpus is too short for the chosen block_size and train_fraction. "
            f"Use a longer corpus (approximately {minimum} characters or more) "
            "or reduce block_size."
        )

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = resolve_device(config.device)
    model_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=config.block_size,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        n_embd=config.n_embd,
        dropout=config.dropout,
    )
    model = TinyTransformerLM(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output_dir / "tokenizer.json")
    history: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    final_train_loss = float("nan")
    final_val_loss = float("nan")

    for step in range(1, config.max_steps + 1):
        model.train()
        x, y = _make_batch(train_data, config.batch_size, config.block_size, device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y)

        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"Invalid training loss at step {step}.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if step % config.eval_interval == 0 or step == config.max_steps:
            final_train_loss = _estimate_loss(model, train_data, config, device)
            final_val_loss = _estimate_loss(model, val_data, config, device)
            record: dict[str, float | int] = {
                "step": step,
                "train_loss": final_train_loss,
                "val_loss": final_val_loss,
            }
            history.append(record)

            checkpoint = {
                "format_version": 1,
                "model_type": "TinyTransformerLM",
                "step": step,
                "model_config": asdict(model_config),
                "training_config": asdict(config),
                "model_state_dict": model.state_dict(),
                "train_loss": final_train_loss,
                "val_loss": final_val_loss,
            }
            torch.save(checkpoint, output_dir / "last.pt")

            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                torch.save(checkpoint, output_dir / "best.pt")

            (output_dir / "metrics.json").write_text(
                json.dumps(history, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"step {step}/{config.max_steps} | "
                f"train_loss={final_train_loss:.4f} | "
                f"val_loss={final_val_loss:.4f} | device={device}"
            )

    return {
        "output_dir": str(output_dir),
        "best_checkpoint": str(output_dir / "best.pt"),
        "last_checkpoint": str(output_dir / "last.pt"),
        "tokenizer_path": str(output_dir / "tokenizer.json"),
        "steps": config.max_steps,
        "best_val_loss": best_val_loss,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "device": str(device),
    }


def load_language_model(
    checkpoint_path: str | Path,
    device: str = "auto",
) -> tuple[TinyTransformerLM, CharacterTokenizer, dict[str, Any]]:
    """Load a checkpoint produced by train_language_model."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    target_device = resolve_device(device)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=target_device, weights_only=True
        )
    except TypeError:
        # Compatibility with older PyTorch versions; only load trusted local files.
        checkpoint = torch.load(checkpoint_path, map_location=target_device)

    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format.")
    if checkpoint.get("model_type") != "TinyTransformerLM":
        raise ValueError("Checkpoint does not contain a TinyTransformerLM.")

    model_config = TransformerConfig(**checkpoint["model_config"])
    tokenizer_path = checkpoint_path.parent / "tokenizer.json"
    tokenizer = CharacterTokenizer.load(tokenizer_path)

    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError("Checkpoint and tokenizer vocabulary sizes do not match.")

    model = TinyTransformerLM(model_config).to(target_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, checkpoint


@torch.no_grad()
def generate_text(
    checkpoint_path: str | Path,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: str = "auto",
) -> str:
    """Generate text from a trained checkpoint, without an external AI API."""
    if not prompt:
        raise ValueError("Prompt cannot be empty.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative.")

    model, tokenizer, _ = load_language_model(checkpoint_path, device=device)
    target_device = next(model.parameters()).device
    input_ids = torch.tensor(
        [tokenizer.encode(prompt)], dtype=torch.long, device=target_device
    )
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(generated[0].tolist())
