import pytest

from ai_platform.machine_learning.language_model.bpe_tokenizer import BPETokenizer


def test_bpe_learns_merges_and_reduces_repeated_word_token_count():
    text = "banana banana banana banana banana"
    tokenizer = BPETokenizer.fit(text, vocab_size=24, min_frequency=2)

    assert tokenizer.merges
    assert len(tokenizer.encode("banana")) < len("banana")
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_round_trip_preserves_english_spaces_punctuation_and_lines():
    text = "Hello, world!\nThis is a BPE tokenizer.\n\nIt preserves spacing."
    tokenizer = BPETokenizer.fit(text, vocab_size=128)

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_round_trip_supports_arabic_and_unicode():
    text = "????? ???????\nAI engineering ? ????? ????"
    tokenizer = BPETokenizer.fit(text, vocab_size=160)

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_encoding_can_add_bos_and_eos():
    tokenizer = BPETokenizer.fit("Alice in Wonderland", vocab_size=64)
    ids = tokenizer.encode("Alice", add_bos=True, add_eos=True)

    assert ids[0] == tokenizer.bos_token_id
    assert ids[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(ids) == "Alice"
    assert tokenizer.decode(ids, skip_special_tokens=False) == "<bos>Alice<eos>"


def test_unknown_character_uses_unknown_token():
    tokenizer = BPETokenizer.fit("abcdef abcdef", vocab_size=32)

    assert tokenizer.encode("?") == [tokenizer.unk_token_id]
    assert tokenizer.decode([tokenizer.unk_token_id]) == ""
    assert tokenizer.decode(
        [tokenizer.unk_token_id], skip_special_tokens=False
    ) == "<unk>"


def test_save_and_load_preserve_merges_and_encoding(tmp_path):
    text = "banana banana\n????? ???????\n"
    tokenizer = BPETokenizer.fit(text, vocab_size=96)
    destination = tmp_path / "bpe" / "tokenizer.json"

    tokenizer.save(destination)
    loaded = BPETokenizer.load(destination)

    assert loaded.tokens == tokenizer.tokens
    assert loaded.merges == tokenizer.merges
    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.decode(loaded.encode(text)) == text


def test_fit_is_deterministic():
    text = "reproducible text reproducible text"
    first = BPETokenizer.fit(text, vocab_size=64)
    second = BPETokenizer.fit(text, vocab_size=64)

    assert first.tokens == second.tokens
    assert first.merges == second.merges


def test_fit_rejects_empty_text():
    with pytest.raises(ValueError, match="empty"):
        BPETokenizer.fit("")


def test_fit_rejects_vocabulary_too_small_for_alphabet():
    with pytest.raises(ValueError, match="too small"):
        BPETokenizer.fit("abc", vocab_size=6)


def test_fit_rejects_invalid_min_frequency():
    with pytest.raises(ValueError, match="min_frequency"):
        BPETokenizer.fit("abcdef", min_frequency=0)


def test_decode_rejects_invalid_token_ids():
    tokenizer = BPETokenizer.fit("abcabc", vocab_size=32)

    with pytest.raises(ValueError, match="out of range"):
        tokenizer.decode([tokenizer.vocab_size])


def test_encode_rejects_non_string_input():
    tokenizer = BPETokenizer.fit("abcabc", vocab_size=32)

    with pytest.raises(TypeError, match="string"):
        tokenizer.encode(123)
