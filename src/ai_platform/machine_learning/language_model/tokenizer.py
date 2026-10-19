"""A deterministic character-level tokenizer trained from supplied text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


class CharacterTokenizer:
    """Learn a character vocabulary directly from a text corpus."""

    SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
    FORMAT_VERSION = 1

    def __init__(self, tokens: Sequence[str]) -> None:
        normalized = tuple(tokens)

        if normalized[:len(self.SPECIAL_TOKENS)] != self.SPECIAL_TOKENS:
            raise ValueError("Vocabulary must begin with the required special tokens.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Vocabulary tokens must be unique.")
        if any(len(token) != 1 for token in normalized[len(self.SPECIAL_TOKENS):]):
            raise ValueError("Every regular vocabulary token must be one character.")

        self.tokens = normalized
        self.token_to_id = {token: index for index, token in enumerate(normalized)}
        self.id_to_token = normalized
        self._special_token_ids = frozenset(
            self.token_to_id[token] for token in self.SPECIAL_TOKENS
        )

    @classmethod
    def fit(cls, text: str) -> "CharacterTokenizer":
        """Build a reproducible vocabulary from the provided text."""
        if not isinstance(text, str):
            raise TypeError("Training text must be a string.")
        if not text:
            raise ValueError("Training text cannot be empty.")

        characters = sorted(set(text))
        return cls((*cls.SPECIAL_TOKENS, *characters))

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["<eos>"]

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Convert text to token IDs; optionally add boundary tokens."""
        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")

        ids = [self.token_to_id.get(char, self.unk_token_id) for char in text]

        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)

        return ids

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Convert token IDs back to text."""
        pieces: list[str] = []

        for token_id in token_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError("Token IDs must be integers.")
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"Token ID out of range: {token_id}")
            if skip_special_tokens and token_id in self._special_token_ids:
                continue
            pieces.append(self.id_to_token[token_id])

        return "".join(pieces)

    def save(self, path: str | Path) -> None:
        """Save the vocabulary and format metadata as UTF-8 JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.FORMAT_VERSION,
            "tokenizer_type": "character",
            "tokens": list(self.tokens),
        }
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        """Restore a tokenizer previously saved with save()."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))

        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("Unsupported tokenizer format version.")
        if payload.get("tokenizer_type") != "character":
            raise ValueError("Saved file is not a character tokenizer.")
        tokens = payload.get("tokens")
        if not isinstance(tokens, list) or not all(
            isinstance(token, str) for token in tokens
        ):
            raise ValueError("Saved tokenizer vocabulary is invalid.")

        return cls(tokens)
