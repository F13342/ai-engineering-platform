"""A small decoder-only Transformer language model initialized from scratch."""

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransformerConfig:
    """Architecture configuration for a decoder-only Transformer."""

    vocab_size: int
    block_size: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in ("vocab_size", "block_size", "n_layers", "n_heads", "n_embd"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")

        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2.")
        if self.block_size < 1:
            raise ValueError("block_size must be positive.")
        if self.n_layers < 1 or self.n_heads < 1 or self.n_embd < 1:
            raise ValueError("Layer count, head count, and embedding size must be positive.")
        if self.n_embd % self.n_heads:
            raise ValueError("n_embd must be divisible by n_heads.")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(self.dropout)
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("dropout must be finite and in the interval [0, 1).")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable configuration."""
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention that cannot attend to future tokens."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.n_embd // config.n_heads

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        mask = torch.triu(
            torch.ones(config.block_size, config.block_size, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, sequence_length, channels = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(
                batch_size,
                sequence_length,
                self.n_heads,
                self.head_dim,
            ).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = self.causal_mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        weights = F.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)
        output = weights @ v

        output = output.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, channels
        )
        return self.residual_dropout(self.proj(output))


class TransformerBlock(nn.Module):
    """Pre-normalized Transformer block with residual connections."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class TinyTransformerLM(nn.Module):
    """A small, randomly initialized autoregressive language model."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._initialize_weights)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Return next-token logits and optional cross-entropy loss."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long token IDs.")

        batch_size, sequence_length = input_ids.shape
        if sequence_length < 1:
            raise ValueError("Input sequences cannot be empty.")
        if sequence_length > self.config.block_size:
            raise ValueError("Input sequence exceeds the configured block_size.")

        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids.")
            if targets.dtype != torch.long:
                raise TypeError("targets must use torch.long token IDs.")

        positions = torch.arange(sequence_length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * sequence_length, self.config.vocab_size),
                targets.reshape(batch_size * sequence_length),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """Generate tokens autoregressively from a prompt of token IDs."""
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must contain a non-empty batch of sequences.")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long token IDs.")
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens must be a non-negative integer.")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("temperature must be finite and positive.")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer or None.")

        was_training = self.training
        self.eval()
        generated = input_ids

        try:
            for _ in range(max_new_tokens):
                context = generated[:, -self.config.block_size:]
                logits, _ = self(context)
                next_logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    k = min(top_k, self.config.vocab_size)
                    threshold = torch.topk(next_logits, k, dim=-1).values[:, -1:]
                    next_logits = next_logits.masked_fill(
                        next_logits < threshold, float("-inf")
                    )

                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
        finally:
            self.train(was_training)

        return generated
