import pytest

from ai_platform.machine_learning.language_model.tokenizer import CharacterTokenizer


def test_fit_creates_deterministic_vocabulary_and_special_ids():
    first = CharacterTokenizer.fit("cabba\n")
    second = CharacterTokenizer.fit("cabba\n")

    assert first.tokens == second.tokens
    assert first.tokens[:4] == ("<pad>", "<unk>", "<bos>", "<eos>")
    assert first.tokens[4:] == tuple(sorted(set("cabba\n")))
    assert first.pad_token_id == 0
    assert first.unk_token_id == 1
    assert first.bos_token_id == 2
    assert first.eos_token_id == 3
    assert first.vocab_size == len(first.tokens)


def test_encode_decode_round_trip_preserves_unicode_and_whitespace():
    text = "?????\nAI 123"
    tokenizer = CharacterTokenizer.fit(text)

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_encode_can_add_beginning_and_end_tokens():
    tokenizer = CharacterTokenizer.fit("ab")

    ids = tokenizer.encode("ab", add_bos=True, add_eos=True)

    assert ids[0] == tokenizer.bos_token_id
    assert ids[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(ids) == "ab"
    assert tokenizer.decode(ids, skip_special_tokens=False) == "<bos>ab<eos>"


def test_unknown_characters_use_unknown_token():
    tokenizer = CharacterTokenizer.fit("abc")

    assert tokenizer.encode("z") == [tokenizer.unk_token_id]
    assert tokenizer.decode([tokenizer.unk_token_id]) == ""
    assert tokenizer.decode(
        [tokenizer.unk_token_id], skip_special_tokens=False
    ) == "<unk>"


def test_save_and_load_preserve_vocabulary_and_encoding(tmp_path):
    text = "????? ?? ?????\n"
    original = CharacterTokenizer.fit(text)
    path = tmp_path / "tokenizers" / "character.json"

    original.save(path)
    restored = CharacterTokenizer.load(path)

    assert restored.tokens == original.tokens
    assert restored.encode(text) == original.encode(text)
    assert restored.decode(restored.encode(text)) == text


def test_fit_rejects_empty_text():
    with pytest.raises(ValueError, match="cannot be empty"):
        CharacterTokenizer.fit("")


def test_decode_rejects_out_of_range_ids():
    tokenizer = CharacterTokenizer.fit("abc")

    with pytest.raises(ValueError, match="out of range"):
        tokenizer.decode([tokenizer.vocab_size])


def test_encode_rejects_non_string_input():
    tokenizer = CharacterTokenizer.fit("abc")

    with pytest.raises(TypeError, match="must be a string"):
        tokenizer.encode(123)
