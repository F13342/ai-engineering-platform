"""Trainable, from-scratch byte-pair-style tokenizer over Unicode characters.

This implementation learns character-pair merges from supplied text. It does
not load pretrained vocabularies, merge tables, or external language models.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Sequence


class BPETokenizer:
    """A deterministic BPE tokenizer trained only on the supplied corpus."""

    SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
    FORMAT_VERSION = 1

    def __init__(
        self,
        tokens: Sequence[str],
        merges: Sequence[Sequence[str]],
    ) -> None:
        normalized_tokens = tuple(tokens)

        if normalized_tokens[:len(self.SPECIAL_TOKENS)] != self.SPECIAL_TOKENS:
            raise ValueError("Vocabulary must begin with the required special tokens.")
        if len(set(normalized_tokens)) != len(normalized_tokens):
            raise ValueError("Vocabulary tokens must be unique.")
        if any(not token for token in normalized_tokens):
            raise ValueError("Vocabulary tokens cannot be empty.")

        token_set = set(normalized_tokens)
        normalized_merges: list[tuple[str, str, str]] = []

        for merge in merges:
            if len(merge) != 3:
                raise ValueError("Every merge must contain left, right, and result.")
            left, right, result = merge
            if left not in token_set or right not in token_set or result not in token_set:
                raise ValueError("Merge references a token missing from the vocabulary.")
            if result != left + right:
                raise ValueError("A merged token must equal left + right.")
            if left in self.SPECIAL_TOKENS or right in self.SPECIAL_TOKENS:
                raise ValueError("Special tokens cannot be BPE merge inputs.")
            normalized_merges.append((left, right, result))

        self.tokens = normalized_tokens
        self.token_to_id = {
            token: index for index, token in enumerate(normalized_tokens)
        }
        self.id_to_token = normalized_tokens
        self.merges = tuple(normalized_merges)
        self._special_token_ids = frozenset(
            self.token_to_id[token] for token in self.SPECIAL_TOKENS
        )

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

    @staticmethod
    def _normalize(text: str) -> str:
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def _split_text(text: str) -> list[str]:
        """Split into letter/number runs, whitespace runs, and punctuation runs."""
        pieces = re.findall(r"\s+|[^\W_]+|[^\w\s]+|_", text, flags=re.UNICODE)
        if "".join(pieces) != text:
            raise ValueError("Text pre-tokenization did not preserve every character.")
        return pieces

    @staticmethod
    def _merge_pair(
        sequence: list[str],
        left: str,
        right: str,
        merged: str,
    ) -> list[str]:
        output: list[str] = []
        i = 0
        while i < len(sequence):
            if (
                i + 1 < len(sequence)
                and sequence[i] == left
                and sequence[i + 1] == right
            ):
                output.append(merged)
                i += 2
            else:
                output.append(sequence[i])
                i += 1
        return output

    @classmethod
    def fit(
        cls,
        text: str,
        vocab_size: int = 512,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """Learn a vocabulary and ordered merge rules from training text."""
        if not isinstance(text, str):
            raise TypeError("Training text must be a string.")
        text = cls._normalize(text)
        if not text:
            raise ValueError("Training text cannot be empty.")
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive.")
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive.")

        pieces = cls._split_text(text)
        alphabet = sorted(set(text))
        initial_tokens = (*cls.SPECIAL_TOKENS, *alphabet)

        if vocab_size < len(initial_tokens):
            raise ValueError(
                "vocab_size is too small for the special tokens and character alphabet. "
                f"Use at least {len(initial_tokens)}."
            )

        vocabulary = list(initial_tokens)
        vocabulary_set = set(vocabulary)
        sequences = [list(piece) for piece in pieces]
        merges: list[tuple[str, str, str]] = []

        while len(vocabulary) < vocab_size:
            pair_counts: dict[tuple[str, str], int] = {}

            for sequence in sequences:
                for i in range(len(sequence) - 1):
                    pair = (sequence[i], sequence[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

            candidates = [
                (pair, frequency)
                for pair, frequency in pair_counts.items()
                if frequency >= min_frequency
                and pair[0] + pair[1] not in vocabulary_set
                and pair[0] not in cls.SPECIAL_TOKENS
                and pair[1] not in cls.SPECIAL_TOKENS
            ]
            if not candidates:
                break

            # Deterministic tie-breaking by token strings.
            (left, right), _ = min(
                candidates,
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
            merged = left + right

            sequences = [
                cls._merge_pair(sequence, left, right, merged)
                for sequence in sequences
            ]
            vocabulary.append(merged)
            vocabulary_set.add(merged)
            merges.append((left, right, merged))

        return cls(vocabulary, merges)

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Convert text into token IDs using the learned merge order."""
        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")
        text = self._normalize(text)
        pieces = self._split_text(text)
        output_ids: list[int] = []

        for piece in pieces:
            sequence = [
                char if char in self.token_to_id else "<unk>"
                for char in piece
            ]

            for left, right, merged in self.merges:
                sequence = self._merge_pair(sequence, left, right, merged)

            output_ids.extend(
                self.token_to_id.get(token, self.unk_token_id)
                for token in sequence
            )

        if add_bos:
            output_ids.insert(0, self.bos_token_id)
        if add_eos:
            output_ids.append(self.eos_token_id)
        return output_ids

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Convert token IDs back to normalized text."""
        pieces: list[str] = []
        for token_id in token_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError("Token IDs must be integers.")
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"Token ID out of range: {token_id}")
            if token_id == self.unk_token_id:
                if skip_special_tokens:
                    pieces.append("\uFFFD")
                else:
                    pieces.append(self.id_to_token[token_id])
                continue
            if skip_special_tokens and token_id in self._special_token_ids:
                continue
            pieces.append(self.id_to_token[token_id])
        return "".join(pieces)

    def save(self, path: str | Path) -> None:
        """Persist vocabulary and ordered merge rules as UTF-8 JSON."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.FORMAT_VERSION,
            "tokenizer_type": "bpe_char",
            "tokens": list(self.tokens),
            "merges": [list(merge) for merge in self.merges],
        }
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load a previously saved BPE tokenizer."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError("Unsupported BPE tokenizer format version.")
        if payload.get("tokenizer_type") != "bpe_char":
            raise ValueError("Saved file is not a BPE tokenizer.")

        tokens = payload.get("tokens")
        merges = payload.get("merges")
        if not isinstance(tokens, list) or not all(
            isinstance(token, str) for token in tokens
        ):
            raise ValueError("Saved vocabulary is invalid.")
        if not isinstance(merges, list) or not all(
            isinstance(merge, list)
            and len(merge) == 3
            and all(isinstance(part, str) for part in merge)
            for merge in merges
        ):
            raise ValueError("Saved merge rules are invalid.")

        return cls(tokens, merges)
