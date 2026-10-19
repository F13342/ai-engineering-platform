"""Language-model components with optional PyTorch loading."""

__all__ = ["BPETokenizer", "CharacterTokenizer", "TinyTransformerLM", "TransformerConfig"]


def __getattr__(name: str):
    if name == "BPETokenizer":
        from .bpe_tokenizer import BPETokenizer
        return BPETokenizer

    if name == "CharacterTokenizer":
        from .tokenizer import CharacterTokenizer
        return CharacterTokenizer

    if name in {"TinyTransformerLM", "TransformerConfig"}:
        from .model import TinyTransformerLM, TransformerConfig
        return {
            "TinyTransformerLM": TinyTransformerLM,
            "TransformerConfig": TransformerConfig,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
